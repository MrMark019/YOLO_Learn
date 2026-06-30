Get-PnpDevice | Select-Object Status, FriendlyName, InstanceId | Where-Object { $_.FriendlyName -like '*webcam*' -or $_.FriendlyName -like '*camera*' -or $_.FriendlyName -like '*ASUS*' }
