[app]
title = Warehouse Scanner
package.name = warehousescanner
package.domain = org.stu3
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx,json
version = 1.0
requirements = python3,kivy,kivymd,openpyxl,pillow
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
