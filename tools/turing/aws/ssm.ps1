# Runs a local bash script on the benchmark box through SSM (AWS-RunShellScript, as root) with arguments, and
# prints its output. The script travels in a quoted heredoc and runs under bash (the document's own shell may be
# sh). SSM has no completion event; `aws ssm wait command-executed` is the CLI's waiter (a poll every 5 s, at most
# 100), so one call waits up to ~8 minutes: a longer step prints its command id, and -CommandId waits again.
# usage: ssm.ps1 -Id i-... -Script box\setup.sh [-ScriptArgs 'a','b'] [-Timeout 3600]   |   ssm.ps1 -Id i-... -CommandId <id>
param([Parameter(Mandatory)][string]$Id, [string]$Script, [string[]]$ScriptArgs = @(), [int]$Timeout = 3600,
      [string]$CommandId)
$r = 'us-east-1'
if (-not $CommandId) {
  $quoted = ($ScriptArgs | ForEach-Object { "'" + $_.Replace("'", "'\''") + "'" }) -join ' '
  $body = [IO.File]::ReadAllLines((Resolve-Path $Script))
  $lines = @('export AWS_DEFAULT_REGION=us-east-1 HOME=/root', "cat > /tmp/wb_step.sh <<'WB_EOF'") + $body +
           @('WB_EOF', "bash /tmp/wb_step.sh $quoted")
  $params = Join-Path $env:TEMP 'whisper-aws-bench-ssm.json'
  [IO.File]::WriteAllText($params, (@{ commands = $lines; executionTimeout = @("$Timeout") } | ConvertTo-Json -Depth 4))
  $CommandId = aws ssm send-command --region $r --instance-ids $Id --document-name AWS-RunShellScript `
    --parameters "file://$params" --query Command.CommandId --output text
}
aws ssm wait command-executed --region $r --command-id $CommandId --instance-id $Id 2>$null
$inv = aws ssm get-command-invocation --region $r --command-id $CommandId --instance-id $Id --output json | ConvertFrom-Json
Write-Output $inv.StandardOutputContent
if ($inv.StandardErrorContent) { Write-Output "--- stderr"; Write-Output $inv.StandardErrorContent }
Write-Output "--- $($inv.Status) (exit $($inv.ResponseCode)) command $CommandId"
