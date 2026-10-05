#include "cuda/decoder_gemm.h"

#include <array>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda/tiled_split_gemm.cuh"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace cuda {

    using TileLaunch = void (*)(const __half*, const __half*, __half*, int, int, int, const SplitGroups&,
                                cudaStream_t);

    // The tiles decoder_gemm_probe.cu measures, by the name CT2_DECODER_TILES gives them.
    static const std::map<std::string, TileLaunch>& tile_launches() {
      static const std::map<std::string, TileLaunch> launches = {
        {"16x64/4", tsg_launch<16, 64, 1>}, {"16x128/4", tsg_launch<16, 128, 1>},
        {"32x32/4", tsg_launch<32, 32>}, {"32x64/4", tsg_launch<32, 64>}, {"64x32/4", tsg_launch<64, 32>},
        {"64x64/4", tsg_launch<64, 64>}, {"32x128/4", tsg_launch<32, 128>}, {"64x128/4", tsg_launch<64, 128>},
        {"64x16/4", tsg_launch<64, 16, 4>}, {"64x16/8", tsg_launch<64, 16, 4, 8>},
        {"32x32/8", tsg_launch<32, 32, 2, 8>}, {"64x32/8", tsg_launch<64, 32, 2, 8>},
        {"16x64/8", tsg_launch<16, 64, 1, 8>},
        {"64x64/6", tsg_launch<64, 64, 2, 6>}, {"64x64/8", tsg_launch<64, 64, 2, 8>},
        {"64x64k64/4", tsg_launch<64, 64, 2, 4, 64>}, {"64x64k64/5", tsg_launch<64, 64, 2, 5, 64>},
        {"64x32k64/6", tsg_launch<64, 32, 2, 6, 64>}, {"32x64k64/6", tsg_launch<32, 64, 2, 6, 64>},
        {"32x32k64/8", tsg_launch<32, 32, 2, 8, 64>},
      };
      return launches;
    }

    enum Kind { qkv, o, ffn1, vocab, ffn2, kinds };

    static int kind_of(dim_t n, dim_t k) {
      if (k == 1280)
        return n == 3840 ? qkv : n == 1280 ? o : n == 5120 ? ffn1 : n == 51866 || n == 51872 ? vocab : -1;
      return n == 1280 && k == 5120 ? ffn2 : -1;
    }

    static const std::array<TileLaunch, kinds>& configured_tiles() {
      static const std::array<TileLaunch, kinds> tiles = [] {
        std::array<TileLaunch, kinds> chosen{};
        const char* env = std::getenv("CT2_DECODER_TILES");
        std::stringstream entries(env ? env : "");
        static const std::map<std::string, int> names = {{"qkv", qkv}, {"o", o}, {"ffn1", ffn1},
                                                         {"vocab", vocab}, {"ffn2", ffn2}};
        for (std::string entry; std::getline(entries, entry, ',');) {
          const auto eq = entry.find('=');
          const auto kind = names.find(entry.substr(0, eq));
          if (eq == std::string::npos || kind == names.end())
            throw std::invalid_argument("CT2_DECODER_TILES: no such product in '" + entry + "'");
          const std::string tile = entry.substr(eq + 1);
          if (tile == "cublas")
            continue;
          const auto launch = tile_launches().find(tile);
          if (launch == tile_launches().end())
            throw std::invalid_argument("CT2_DECODER_TILES: no such tile in '" + entry + "'");
          chosen[kind->second] = launch->second;
        }
        return chosen;
      }();
      return tiles;
    }

    bool decoder_gemm(const float16_t* a, const float16_t* w, float16_t* c, dim_t m, dim_t n, dim_t k,
                      const std::vector<dim_t>& group_rows, cudaStream_t stream) {
      static const bool verified = cublas_verified_on(8, 9);   // the bits are this device's and cuBLAS build's
      const int kind = kind_of(n, k);
      if (!verified || kind < 0 || m < 2 || m > gsg_max_rows || group_rows.empty()
          || group_rows.size() > static_cast<size_t>(gsg_max_groups) || !configured_tiles()[kind])
        return false;
      SplitGroups groups{};
      int rows = 0;
      for (const dim_t group : group_rows) {
        int slice = static_cast<int>(k), slices = 1;         // one chain over k
        if (kind == ffn2 && !gsg_split_of(group, slice, slices))
          return false;
        rows += static_cast<int>(group);
        groups.row_end[groups.count] = rows;
        groups.slice[groups.count] = slice;
        groups.slices[groups.count] = slices;
        ++groups.count;
      }
      if (rows != m)
        return false;
      configured_tiles()[kind](reinterpret_cast<const __half*>(a), reinterpret_cast<const __half*>(w),
                               reinterpret_cast<__half*>(c), static_cast<int>(m), static_cast<int>(n),
                               static_cast<int>(k), groups, stream);
      return true;
    }

  }
}
