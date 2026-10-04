# A throwaway GPU box for the AWS Whisper benchmark (README.md). Deep Learning Base AMI (Ubuntu 24.04 with the
# NVIDIA driver), the docvoice-ec2-ssm profile (SSM + the code bucket), tagged Project=whisper-aws-bench, in the
# AZ asked for. It terminates itself after -Minutes even if nobody is watching: shutdown behaviour "terminate"
# and a shutdown timer in its user data. Prints the instance id.
param([string]$Type = 'g6e.xlarge', [string]$Az = 'us-east-1a', [int]$Minutes = 150, [int]$DiskGb = 100,
      [string]$Region = 'us-east-1')
$r = $Region   # no ErrorActionPreference=Stop: PS 5.1 would die on the CLI's first stderr line, unread
$ami = aws ssm get-parameter --region $r --query Parameter.Value --output text `
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id
$rootDev = aws ec2 describe-images --region $r --image-ids $ami --query 'Images[0].RootDeviceName' --output text
# -Az '': no subnet, so EC2 picks any AZ of the default VPC that has the capacity right now.
$subnet = if ($Az) { aws ec2 describe-subnets --region $r --query 'Subnets[0].SubnetId' --output text `
  --filters Name=default-for-az,Values=true "Name=availability-zone,Values=$Az" } else { $null }
$tags = @(@{ Key = 'Name'; Value = 'whisper-aws-bench' }, @{ Key = 'Project'; Value = 'whisper-aws-bench' })
$ud = Join-Path $env:TEMP 'whisper-aws-bench-userdata.sh'
[IO.File]::WriteAllText($ud, "#!/bin/bash`nshutdown -h +$Minutes`n")
$spec = @{
  ImageId = $ami; InstanceType = $Type; MinCount = 1; MaxCount = 1
  IamInstanceProfile = @{ Name = 'docvoice-ec2-ssm' }; InstanceInitiatedShutdownBehavior = 'terminate'
  BlockDeviceMappings = @(@{ DeviceName = $rootDev; Ebs = @{ VolumeSize = $DiskGb; VolumeType = 'gp3'; DeleteOnTermination = $true } })
  TagSpecifications = @(@{ ResourceType = 'instance'; Tags = $tags }, @{ ResourceType = 'volume'; Tags = $tags })
}
if ($subnet) { $spec.SubnetId = $subnet }
$json = Join-Path $env:TEMP 'whisper-aws-bench-launch.json'
[IO.File]::WriteAllText($json, ($spec | ConvertTo-Json -Depth 6))
$res = aws ec2 run-instances --region $r --cli-input-json "file://$json" --user-data "file://$ud" `
  --query 'Instances[0].InstanceId' --output text 2>&1
$id = "$res".Trim()
if ($LASTEXITCODE -ne 0 -or $id -notmatch '^i-[0-9a-f]+$') { throw "run-instances failed: $res" }
Write-Output "launched $id ($Type, $Az, $ami, self-terminates after $Minutes min)"
aws ec2 wait instance-running --region $r --instance-ids $id
# SSM has no event for "agent registered": poll it (about 1-2 minutes after boot).
for ($i = 0; $i -lt 40; $i++) {
  $ping = aws ssm describe-instance-information --region $r --filters "Key=InstanceIds,Values=$id" `
    --query 'InstanceInformationList[0].PingStatus' --output text
  if ($ping -eq 'Online') { Write-Output "$id online in SSM"; Write-Output $id; exit 0 }
  Start-Sleep -Seconds 10
}
throw "$id never came online in SSM"
