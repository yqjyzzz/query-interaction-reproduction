param(
    [Parameter(Mandatory = $true)]
    [string]$Builder,
    [string]$Config = "cv_demo/configs/demo.json"
)

python -m cv_demo.inference.predict --config $Config --builder $Builder
