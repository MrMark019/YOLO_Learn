# Check for any camera or video devices
Write-Host "=== Camera/Video Devices ==="
Get-PnpDevice | Where-Object {
    $_.FriendlyName -like '*camera*' -or
    $_.FriendlyName -like '*webcam*' -or
    $_.FriendlyName -like '*video*' -or
    $_.FriendlyName -like '*ASUS FHD*'
} | Select-Object Status, FriendlyName, InstanceId | Format-Table -AutoSize

# Check USB devices with VID_3277 (ASUS camera vendor)
Write-Host "`n=== USB Devices with ASUS Camera VID ==="
Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_3277*' } | Select-Object Status, FriendlyName, InstanceId | Format-Table -AutoSize

# Check for any Unknown USB devices
Write-Host "`n=== Unknown USB Devices ==="
Get-PnpDevice -Class USB | Where-Object { $_.Status -eq 'Unknown' -or $_.Status -eq 'Error' } | Select-Object Status, FriendlyName, InstanceId | Format-Table -AutoSize
