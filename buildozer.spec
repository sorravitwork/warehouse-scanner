[app]
title = Warehouse Scanner
package.name = warehousescanner
package.domain = org.yourname

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx,json

# Your main filebuildozer android debug
main = Scanner_Kivy_Mobile

version = 1.0

# ⚠️ These requirements match your imports
requirements = python3,kivy,kivymd,opencv,numpy,openpyxl,pillow,pyzbar

# Permissions needed for camera + storage
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

android.api = 33
android.minapi = 24
android.ndk = 25b
android.arch = arm64-v8a

[buildozer]
log_level = 2
