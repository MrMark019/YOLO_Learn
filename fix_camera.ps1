# Remove the DFU device
Write-Host "Removing Camera DFU Device..."
pnputil /remove-device "USB\VID_3277&PID_0059&MI_05\7&17E4D989&0&0005"

# Scan for hardware changes
Write-Host "`nScanning for hardware changes..."
pnputil /scan-devices

# Check if camera is detected
Write-Host "`nChecking camera status..."
$camera = Get-PnpDevice | Where-Object { $_.FriendlyName -like '*webcam*' -or $_.FriendlyName -like '*ASUS FHD*' }
if ($camera) {
    Write-Host "Camera Found: $($camera.FriendlyName) - Status: $($camera.Status)"
} else {
    Write-Host "Camera not detected after scan"
}

# Check for any DFU devices
$dfu = Get-PnpDevice | Where-Object { $_.FriendlyName -like '*DFU*' }
if ($dfu) {
    Write-Host "`nDFU Device still present: $($dfu.FriendlyName) - Status: $($dfu.Status)"
}
