"""What the whisper-bench jobs of a day ran, in every region with a stack: each job's time from start to stop, and
the on-demand price of its instance for that time plus a boot and teardown allowance (Batch bills the instance, not
the job). Cost Explorer has no breakdown here (the Project tag is not a cost allocation tag).
usage: python spend.py <YYYY-MM-DD local day> [--overhead-min 6]"""
import argparse, datetime as dt
from settings import FLEET_ALSO, FLEETS, client, fleet_name

PRICE = {"g6e.xlarge": 1.861, "g6e.2xlarge": 2.242,       # us-east-1 on-demand, USD an hour
         "g7e.2xlarge": 3.363, "g7e.4xlarge": 3.998}
# g6e.xlarge on-demand elsewhere (the Pricing API, 5.10.2026); the other types at their us-east-1 price there.
PRICE_IN = {("eu-north-1", "g6e.xlarge"): 1.974, ("eu-central-1", "g6e.xlarge"): 2.327,
            ("ap-south-1", "g6e.xlarge"): 2.235, ("ap-northeast-2", "g6e.xlarge"): 2.288}
REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-north-1", "eu-central-1", "ap-south-1", "ap-northeast-2"]
p = argparse.ArgumentParser(); p.add_argument("day"); p.add_argument("--overhead-min", type=float, default=6)
a = p.parse_args()
day = dt.datetime.strptime(a.day, "%Y-%m-%d")
total, hours = 0.0, 0.0
for region in REGIONS:
    batch = client("batch", region)
    for fleet, (itype, _, _) in FLEETS.items():
        try:
            jobs = []
            for status in ("SUCCEEDED", "FAILED"):
                kw = {"jobQueue": fleet_name(fleet), "jobStatus": status}
                while True:
                    r = batch.list_jobs(**kw)
                    jobs += r["jobSummaryList"]
                    if not r.get("nextToken"):
                        break
                    kw["nextToken"] = r["nextToken"]
        except Exception:
            continue
        for j in jobs:
            if not j.get("startedAt"):
                continue                                     # cancelled before an instance ran it
            start = dt.datetime.fromtimestamp(j["startedAt"] / 1000)
            if start.date() != day.date():
                continue
            stop = dt.datetime.fromtimestamp(j.get("stoppedAt", j["startedAt"]) / 1000)
            h = (stop - start).total_seconds() / 3600 + a.overhead_min / 60
            # A fleet that may take larger sizes: the dearest (the job record does not name the instance).
            cost = h * max(PRICE_IN.get((region, t), PRICE[t]) for t in [itype, *FLEET_ALSO.get(fleet, [])])
            total += cost; hours += h
            print(f"{region:13s} {itype:12s} {j['jobName'][:24]:24s} {start:%H:%M}-{stop:%H:%M} {h:5.2f} h ${cost:5.2f}")
print(f"total {hours:.2f} instance-hours, ${total:.2f}")
