[app]
title = Warehouse Scanner
package.name = warehousescanner
package.domain = org.yourname

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx,json

version = 1.0

requirements = python3,kivy,kivymd,opencv,openpyxl,pillow,pyzbar

android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a

android.accept_sdk_license = True

[buildozer]
log_level = 2
