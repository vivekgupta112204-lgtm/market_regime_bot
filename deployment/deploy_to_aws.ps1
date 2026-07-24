# AWS 1-Click Deployment Script for Market Regime Trading Bot
# This script provisions the CloudFormation Stack and pushes code to EC2

$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "🚀 INITIALIZING AWS PRODUCTION DEPLOYMENT SEQUENCE" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Check if AWS CLI is installed
if (!(Get-Command aws -ErrorAction SilentlyContinue | Out-Null)) {
    # Try adding to path just in case
    $env:PATH += ";C:\Users\vivek\AppData\Local\Programs\Amazon\AWSCLIV2"
}

Write-Host "`n[1] Verifying AWS Identity..." -ForegroundColor Yellow
$identity = aws sts get-caller-identity | ConvertFrom-Json
if ($identity) {
    Write-Host "✅ Logged in as: $($identity.Arn)" -ForegroundColor Green
} else {
    Write-Host "❌ AWS CLI is not authenticated. Please run 'aws login' first." -ForegroundColor Red
    exit
}

$StackName = "InstitutionalTradingBotStack"
$TemplatePath = "aws-infrastructure.yml"

Write-Host "`n[2] Deploying Infrastructure (CloudFormation)..." -ForegroundColor Yellow
Write-Host "This process provisions your 24/7 EC2 Server, Security Groups, and IAM Roles."

try {
    # Test if we have full AWS privileges (checks for OptInRequired blocks)
    $vpcCheck = aws ec2 describe-vpcs --region us-east-1 2>&1
    if ($vpcCheck -match "OptInRequired") {
        Write-Host "`n❌ AWS ERROR: Your billing/credit card is not fully activated yet." -ForegroundColor Red
        Write-Host "Please complete registration at aws.amazon.com, then rerun this script." -ForegroundColor Red
        exit
    }

    # Execute CloudFormation Stack Deployment
    aws cloudformation deploy `
        --template-file $TemplatePath `
        --stack-name $StackName `
        --capabilities CAPABILITY_NAMED_IAM `
        --region us-east-1

    Write-Host "`n✅ Infrastructure Deployment Complete!" -ForegroundColor Green

    # Get the Public IP of the deployed bot
    Write-Host "`n[3] Fetching Bot Server IP Address..." -ForegroundColor Yellow
    $ip = aws cloudformation describe-stacks `
        --stack-name $StackName `
        --query "Stacks[0].Outputs[?OutputKey=='PublicIP'].OutputValue" `
        --output text --region us-east-1

    Write-Host "📡 BOT IS LIVE AT PUBLIC IP: $ip" -ForegroundColor Cyan
    Write-Host "`nTo SSH into your bot (or use AWS Session Manager):"
    Write-Host "aws ssm start-session --target <InstanceID>"

} catch {
    Write-Host "`n❌ Deployment Failed: $_" -ForegroundColor Red
}

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "Deployment Pipeline Finished." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
