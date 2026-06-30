$device = Get-PnpDevice | Where-Object { $_.FriendlyName -like '*DFU*' }
if ($device) {
    Write-Host "DFU Device Found:"
    Write-Host "Status: $($device.Status)"
    Write-Host "FriendlyName: $($device.FriendlyName)"
    Write-Host "InstanceId: $($device.InstanceId)"

    # Try to get more properties
    $props = $device | Get-PnpDeviceProperty
    foreach ($prop in $props) {
        if ($prop.Data -ne $null -and $prop.Data -ne "") {
            Write-Host "$($prop.KeyName): $($prop.Data)"
        }
    }
}
