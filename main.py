"""
╔══════════════════════════════════════════════════════════════╗
║   WAREHOUSE SCANNER SYSTEM  —  MOBILE VERSION (Kivy)         ║
║   Android / iOS compatible via Kivy + KivyMD                 ║
║   Face Recognition with OpenCV YuNet                         ║
║   UPGRADED: Full-Screen High-Contrast Barcode Scanner        ║
╚══════════════════════════════════════════════════════════════╝

REQUIREMENTS (PC dev / Buildozer):
    pip install kivy kivymd opencv-python opencv-contrib-python
    pip install pillow openpyxl numpy pyzbar

    For Android build, use Buildozer:
        pip install buildozer
        buildozer android debug

FILES NEEDED:
    face_detection_yunet_2023mar.onnx

NOTES:
    • Barcode scanning uses the camera (ZBar/pyzbar).
    • Full-screen frame processing with automatic contrast enhancement.
    • Thai Buddhist calendar (พ.ศ.) is supported.
"""

import os, sys, json, datetime, threading
import numpy as np

# ── Kivy config (must be BEFORE kivy imports) ──
os.environ.setdefault("KIVY_NO_ENV_CONFIG", "0")

from kivy.config import Config
Config.set("graphics", "width",  "400")
Config.set("graphics", "height", "780")
Config.set("input", "mouse", "mouse,disable_multitouch")

from kivy.app        import App
from kivy.clock      import Clock
from kivy.uix.boxlayout    import BoxLayout
from kivy.uix.label        import Label
from kivy.uix.button       import Button
from kivy.uix.textinput    import TextInput
from kivy.uix.popup        import Popup
from kivy.uix.image        import Image as KivyImage
from kivy.uix.widget       import Widget
from kivy.graphics.texture import Texture
from kivy.graphics         import Color, RoundedRectangle
from kivy.metrics          import dp
from kivy.lang             import Builder

import cv2

# pyzbar decodes Code 128 / Code 39 / QR and all common 1D barcodes from camera
try:
    from pyzbar import pyzbar as _pyzbar
    PYZBAR_OK = True
except ImportError:
    PYZBAR_OK = False   # graceful fallback — keyboard input still works

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils  import get_column_letter

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
EMPLOYEE_FILE = "employee_database.json"
FACE_DIR      = "employee_faces_dataset"
YUNET_MODEL   = "face_detection_yunet_2023mar.onnx"

DEFAULT_EMPLOYEES = {"EMP-001": "A", "EMP-002": "B", "EMP-003": "C"}

PRODUCTS = {
    "P-KIM":   {"name": "กิมจิ",       "unit": "ถุง"},
    "P-SEA":   {"name": "ยำสาหร่าย",   "unit": "ถุง"},
    "P-KIMST": {"name": "กิมจิ-SENTA", "unit": "ถุง"},
}
MODE_BARCODES = {"MODE-KIM": "P-KIM", "MODE-SEA": "P-SEA", "MODE-KIMST": "P-KIMST"}
MONTH_THAI = {
    1:"มกราคม", 2:"กุมภาพันธ์", 3:"มีนาคม", 4:"เมษายน",
    5:"พฤษภาคม", 6:"มิถุนายน", 7:"กรกฎาคม", 8:"สิงหาคม",
    9:"กันยายน", 10:"ตุลาคม", 11:"พฤศจิกายน", 12:"ธันวาคม",
}

# ─── Color palette ─────────────────────────
BG      = (0.059, 0.067, 0.090, 1)
SURFACE = (0.102, 0.114, 0.153, 1)
SURF2   = (0.145, 0.157, 0.212, 1)
ACCENT  = (0.310, 0.557, 0.969, 1)
GREEN   = (0.220, 0.851, 0.588, 1)
RED     = (0.969, 0.361, 0.361, 1)
GOLD    = (0.969, 0.788, 0.282, 1)
WHITE   = (0.910, 0.918, 0.965, 1)
MUTED   = (0.478, 0.498, 0.604, 1)

# ─────────────────────────────────────────────
#  EMPLOYEE HELPERS
# ─────────────────────────────────────────────
os.makedirs(FACE_DIR, exist_ok=True)

def load_employees():
    if os.path.exists(EMPLOYEE_FILE):
        try:
            with open(EMPLOYEE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_EMPLOYEES.copy()

def save_employees(data):
    with open(EMPLOYEE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

EMPLOYEES = load_employees()

# ─────────────────────────────────────────────
#  OPENCV MODEL
# ─────────────────────────────────────────────
yn_detector     = None
face_recognizer = None

def init_models():
    global yn_detector, face_recognizer
    if not os.path.exists(YUNET_MODEL):
        return False
    yn_detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL, config="",
        input_size=(320, 240),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000,
    )
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    return True

def train_model():
    faces, labels, label_map = [], [], {}
    for idx, code in enumerate(EMPLOYEES):
        folder = os.path.join(FACE_DIR, code)
        if not os.path.exists(folder):
            continue
        for fn in os.listdir(folder):
            if fn.endswith(".jpg"):
                img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(cv2.equalizeHist(img))
                    labels.append(idx)
                    label_map[idx] = code
    if faces and face_recognizer:
        face_recognizer.train(faces, np.array(labels))
        return label_map
    return None

# ─────────────────────────────────────────────
#  EXCEL EXPORT
# ─────────────────────────────────────────────
def save_transaction(doc_num, emp_name, item_id, qty):
    now        = datetime.datetime.now()
    year_th    = now.year + 543
    date_str   = f"{now.strftime('%d-%m')}-{year_th}"
    time_str   = now.strftime("%H:%M:%S")
    month_name = MONTH_THAI[now.month]

    year_dir  = str(year_th)
    os.makedirs(year_dir, exist_ok=True)
    file_path = os.path.join(year_dir, f"{month_name} {year_th}.xlsx")

    wb = openpyxl.load_workbook(file_path) if os.path.exists(file_path) else openpyxl.Workbook()
    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    if "INDEX_วันที่" not in wb.sheetnames:
        ws_idx = wb.create_sheet("INDEX_วันที่", 0)
    else:
        ws_idx = wb["INDEX_วันที่"]

    prod_name  = PRODUCTS[item_id]["name"]
    prod_unit  = PRODUCTS[item_id]["unit"]
    sheet_day  = f"วันที่_{date_str}"
    sheet_data = f"DATA_{doc_num}"

    if sheet_data not in wb.sheetnames:
        ws_data = wb.create_sheet(sheet_data)
        ws_data.append([f"↩️ กลับ ➔ {sheet_day}", *[""] * 7])
        ws_data.append(["วันที่","เวลา","เลขที่ใบจัด","พนักงาน","รหัสสินค้า","สินค้า","จำนวน","หน่วย"])
    else:
        ws_data = wb[sheet_data]

    ws_data.append([date_str, time_str, str(doc_num), emp_name,
                    item_id, prod_name, int(qty), prod_unit])
    ws_data.cell(1, 1).hyperlink = f"#'{sheet_day}'!A2"

    if sheet_day not in wb.sheetnames:
        ws_day = wb.create_sheet(sheet_day)
        ws_day.append(["↩️ INDEX_วันที่", ""])
        ws_day.append(["เลขที่ใบจัด", "รายละเอียด"])
    else:
        ws_day = wb[sheet_day]
    ws_day.cell(1, 1).hyperlink = "#'INDEX_วันที่'!A1"

    if not any(str(ws_day.cell(r, 1).value) == str(doc_num)
               for r in range(3, ws_day.max_row + 1)):
        row = ws_day.max_row + 1
        ws_day.cell(row, 1).value = str(doc_num)
        lc = ws_day.cell(row, 2)
        lc.value     = "🔗 รายละเอียด"
        lc.hyperlink = f"#'{sheet_data}'!A2"
        lc.font      = Font(name="Segoe UI", color="0563C1", underline="single")

    if ws_idx.max_row == 1 and ws_idx.cell(1, 1).value is None:
        ws_idx.append(["วันที่", "ลิงก์"])
    if not any(str(ws_idx.cell(r, 1).value) == date_str
               for r in range(2, ws_idx.max_row + 1)):
        row = ws_idx.max_row + 1
        ws_idx.cell(row, 1).value = date_str
        lc = ws_idx.cell(row, 2)
        lc.value     = f"📅 {date_str}"
        lc.hyperlink = f"#'{sheet_day}'!A2"
        lc.font      = Font(name="Segoe UI", color="0563C1", underline="single")

    # Styling
    hdr_fill = PatternFill("solid", fgColor="1F4E78")
    even_fill= PatternFill("solid", fgColor="F2F7FA")
    odd_fill = PatternFill("solid", fgColor="FFFFFF")
    hdr_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    dat_font = Font(name="Segoe UI", size=11, color="333333")
    center   = Alignment(horizontal="center", vertical="center")
    thin     = Border(**{s: Side(style="thin", color="D9D9D9")
                         for s in ("left","right","top","bottom")})

    for ws in wb.worksheets:
        is_sub  = ws.title != "INDEX_วันที่"
        hdr_row = 2 if is_sub else 1
        start   = 3 if is_sub else 2
        ws.row_dimensions[hdr_row].height = 28
        for cell in ws[hdr_row]:
            cell.fill = hdr_fill; cell.font = hdr_font
            cell.alignment = center; cell.border = thin
        for row in range(start, ws.max_row + 1):
            ws.row_dimensions[row].height = 22
            rfill = even_fill if row % 2 == 0 else odd_fill
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row, col)
                cell.border = thin
                if not cell.hyperlink:
                    cell.fill = rfill; cell.font = dat_font
                cell.alignment = center
        for col in ws.columns:
            mlen = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = max(mlen+6, 18)
        ws.views.sheetView[0].showGridLines = True

    wb.save(file_path)

# ─────────────────────────────────────────────
#  KV LAYOUT STRING
# ─────────────────────────────────────────────
KV = """
#:import dp kivy.metrics.dp

<RoundBtn@Button>:
    background_normal: ''
    background_color: 0.310, 0.557, 0.969, 1
    color: 1, 1, 1, 1
    font_size: dp(15)
    size_hint_y: None
    height: dp(52)
    bold: True

<DarkLabel@Label>:
    color: 0.910, 0.918, 0.965, 1
    font_size: dp(14)

<DarkInput@TextInput>:
    background_color: 0.145, 0.157, 0.212, 1
    foreground_color: 0.910, 0.918, 0.965, 1
    cursor_color: 0.310, 0.557, 0.969, 1
    font_size: dp(16)
    multiline: False
    size_hint_y: None
    height: dp(50)
    padding: [dp(12), dp(12)]

<StatusBanner@Label>:
    font_size: dp(16)
    bold: True
    size_hint_y: None
    height: dp(70)
    text_size: self.width, None
    halign: 'center'
    valign: 'middle'
"""

# ─────────────────────────────────────────────
#  FACE LOGIN SCREEN
# ─────────────────────────────────────────────
class FaceLoginPopup(Popup):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.title          = "📸 Face Login"
        self.size_hint      = (0.98, 0.92)
        self.auto_dismiss   = False
        self.callback       = callback
        self.label_map      = train_model()
        self.consecutive    = 0
        self.last_code      = None
        self.cap            = None
        self._event         = None

        layout = BoxLayout(orientation="vertical", padding=dp(10), spacing=dp(8))

        self.cam_img = KivyImage(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.cam_img)

        self.status = Label(
            text="⏳ Scanning...",
            font_size=dp(14), bold=True,
            color=(0.969, 0.788, 0.282, 1),
            size_hint_y=None, height=dp(36))
        layout.add_widget(self.status)

        close_btn = Button(
            text="✕ Cancel", size_hint_y=None, height=dp(44),
            background_color=(0.969, 0.361, 0.361, 1),
            background_normal="", bold=True)
        close_btn.bind(on_release=self._close)
        layout.add_widget(close_btn)

        self.content = layout
        self.bind(on_open=self._start)

    def _start(self, *_):
        self.cap    = cv2.VideoCapture(0)
        self._event = Clock.schedule_interval(self._update, 1/20)

    def _update(self, dt):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if yn_detector:
            yn_detector.setInputSize((w, h))
            _, faces = yn_detector.detect(frame)
            if faces is not None and self.label_map:
                for face in faces:
                    x, y, wb_, hb = map(int, face[:4])
                    pw, ph = int(wb_*.15), int(hb*.15)
                    xp = max(0, x-pw); yp = max(0, y-ph)
                    wp = min(w-xp, wb_+pw*2); hp = min(h-yp, hb+ph*2)
                    if wp <= 0 or hp <= 0:
                        continue
                    cv2.rectangle(frame, (xp,yp), (xp+wp,yp+hp), (79,142,247), 2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi  = gray[yp:yp+hp, xp:xp+wp]
                    if roi.size == 0:
                        continue
                    roi = cv2.equalizeHist(cv2.resize(roi, (200,200)))
                    lbl, conf = face_recognizer.predict(roi)
                    if conf < 55:
                        code = self.label_map[lbl]
                        name = EMPLOYEES[code]
                        cv2.putText(frame, f"{name} ({int(conf)})",
                                    (xp, yp-8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (56,217,150), 2)
                        if code == self.last_code:
                            self.consecutive += 1
                        else:
                            self.last_code   = code
                            self.consecutive = 1
                        self.status.text = (
                            f"🔍 Verifying {name}  [{self.consecutive}/12]")
                        self.status.color = (0.22, 0.85, 0.59, 1)
                        if self.consecutive >= 12:
                            self._close()
                            self.callback(code)
                            return
                    else:
                        cv2.putText(frame, "Unknown",
                                    (xp, yp-8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (247,92,92), 2)
                        self.consecutive = 0
                        self.status.text  = "❓ Unknown face"
                        self.status.color = (0.97, 0.36, 0.36, 1)

        # Display frame
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb  = cv2.flip(rgb, 0)
        tex  = Texture.create(size=(w, h), colorfmt="rgb")
        tex.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.cam_img.texture = tex

    def _close(self, *_):
        if self._event:
            self._event.cancel()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.dismiss()

# ─────────────────────────────────────────────
#  EMPLOYEE MANAGER POPUP
# ─────────────────────────────────────────────
class EmployeeManagerPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title      = "⚙️ Employee Manager"
        self.size_hint  = (0.96, 0.88)

        layout = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(12))

        layout.add_widget(Label(
            text="Staff Management",
            font_size=dp(20), bold=True,
            color=(0.91,0.92,0.97,1),
            size_hint_y=None, height=dp(40)))

        def mk_btn(txt, col, cb):
            b = Button(text=txt, background_color=col, background_normal="",
                       font_size=dp(14), bold=True,
                       size_hint_y=None, height=dp(52))
            b.bind(on_release=lambda *_: cb())
            layout.add_widget(b)

        mk_btn("📋  View All Employees",         (0.31,0.56,0.97,1), self._show_list)
        mk_btn("🥳  Add Employee + Face Capture",(0.22,0.85,0.59,1), self._add_emp)
        mk_btn("😰  Remove Employee",            (0.97,0.36,0.36,1), self._del_emp)
        mk_btn("✕  Close",                      (0.18,0.19,0.26,1), self.dismiss)

        self.content = layout

    def _show_list(self):
        p = Popup(title="Employee List", size_hint=(0.96, 0.85))
        from kivy.uix.scrollview import ScrollView
        sv = ScrollView()
        col = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4),
                        padding=dp(10))
        col.bind(minimum_height=col.setter("height"))

        for code, name in EMPLOYEES.items():
            folder = os.path.join(FACE_DIR, code)
            cnt = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) \
                  if os.path.exists(folder) else 0
            row = BoxLayout(size_hint_y=None, height=dp(50),
                            padding=(dp(10),0))
            row.add_widget(Label(
                text=f"[b]{code}[/b]  —  {name}\n[color=888888]{'✅' if cnt else '❌'} {cnt} images[/color]",
                markup=True, halign="left", valign="middle",
                color=(0.91,0.92,0.97,1), font_size=dp(13)))
            col.add_widget(row)

        sv.add_widget(col)
        p.content = sv
        p.open()

    def _add_emp(self):
        p = Popup(title="Register Employee", size_hint=(0.97, 0.95),
                  auto_dismiss=False)
        TARGET = 50
        state = {"cap": None, "event": None, "capturing": False, "count": 0}

        layout = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(8))
        layout.add_widget(Label(text="Employee ID:", font_size=dp(13), color=(0.48,0.50,0.60,1), size_hint_y=None, height=dp(24)))
        ent_code = TextInput(hint_text="e.g. EMP-004", background_color=SURF2, foreground_color=WHITE, font_size=dp(15), multiline=False, size_hint_y=None, height=dp(44))
        layout.add_widget(ent_code)

        layout.add_widget(Label(text="Full Name:", font_size=dp(13), color=(0.48,0.50,0.60,1), size_hint_y=None, height=dp(24)))
        ent_name = TextInput(hint_text="First Last", background_color=SURF2, foreground_color=WHITE, font_size=dp(15), multiline=False, size_hint_y=None, height=dp(44))
        layout.add_widget(ent_name)

        cam_img = KivyImage(allow_stretch=True, keep_ratio=True)
        layout.add_widget(cam_img)

        prog_lbl = Label(text="0 / 50  —  tap Start to begin", font_size=dp(13), bold=True, color=(0.31,0.56,0.97,1), size_hint_y=None, height=dp(30))
        layout.add_widget(prog_lbl)

        def start_cap(*_):
            if not ent_code.text.strip() or not ent_name.text.strip():
                prog_lbl.text  = "⚠️  Fill in ID and Name first!"
                prog_lbl.color = (0.97,0.36,0.36,1)
                return
            state["count"]     = 0
            state["capturing"] = True
            prog_lbl.color     = (0.22,0.85,0.59,1)

        btn_start = Button(text="🚀 Start Face Capture  (turn head slowly)", background_color=(0.22,0.85,0.59,1), background_normal="", bold=True, font_size=dp(14), size_hint_y=None, height=dp(52))
        btn_start.bind(on_release=start_cap)
        layout.add_widget(btn_start)

        def cancel(*_):
            if state["event"]: state["event"].cancel()
            if state["cap"] and state["cap"].isOpened(): state["cap"].release()
            p.dismiss()

        btn_close = Button(text="✕ Cancel", background_color=(0.18,0.19,0.26,1), background_normal="", font_size=dp(13), size_hint_y=None, height=dp(44))
        btn_close.bind(on_release=cancel)
        layout.add_widget(btn_close)

        def upd_cam(dt):
            cap = state["cap"]
            if not cap or not cap.isOpened(): return
            ret, frame = cap.read()
            if not ret: return

            frame = cv2.flip(frame, 1)
            h, w_ = frame.shape[:2]

            if yn_detector:
                yn_detector.setInputSize((w_, h))
                _, faces = yn_detector.detect(frame)
                if faces is not None:
                    for face in faces:
                        x, y, wb_, hb = map(int, face[:4])
                        pw, ph = int(wb_ * .15), int(hb * .15)
                        xp = max(0, x - pw);  yp = max(0, y - ph)
                        wp = min(w_ - xp, wb_ + pw * 2)
                        hp = min(h  - yp, hb  + ph * 2)
                        if wp <= 0 or hp <= 0: continue
                        cv2.rectangle(frame, (xp,yp), (xp+wp,yp+hp), (56,217,150), 2)

                        if state["capturing"]:
                            code   = ent_code.text.strip().upper()
                            folder = os.path.join(FACE_DIR, code)
                            os.makedirs(folder, exist_ok=True)

                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            crop = gray[yp:yp+hp, xp:xp+wp]
                            if crop.size == 0: continue

                            state["count"] += 1
                            cv2.imwrite(os.path.join(folder, f"{state['count']}.jpg"), cv2.resize(crop, (200, 200)))
                            prog_lbl.text = f"{state['count']} / {TARGET}"

                            if state["count"] >= TARGET:
                                state["capturing"] = False
                                EMPLOYEES[code] = ent_name.text.strip()
                                save_employees(EMPLOYEES)
                                if state["event"]: state["event"].cancel()
                                cap.release()
                                p.dismiss()
                                Popup(title="✅ Done", content=Label(text=f"Registered {TARGET} face samples\nfor {EMPLOYEES[code]}!", halign="center"), size_hint=(.75, .3)).open()
                                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.flip(rgb, 0)
            tex = Texture.create(size=(w_, h), colorfmt="rgb")
            tex.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            cam_img.texture = tex

        p.content = layout

        def on_open(*_):
            for idx in (0, 1, 2):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    state["cap"] = cap
                    break
            if not state["cap"]:
                prog_lbl.text  = "❌  No camera found!"
                prog_lbl.color = (0.97, 0.36, 0.36, 1)
                return
            state["event"] = Clock.schedule_interval(upd_cam, 1 / 20)

        p.bind(on_open=on_open)
        p.open()

    def _del_emp(self):
        p = Popup(title="Remove Employee", size_hint=(.9,.4))
        v = BoxLayout(orientation="vertical", padding=dp(16), spacing=dp(10))
        v.add_widget(Label(text="Enter Employee ID to remove:", color=(0.91,0.92,0.97,1)))
        ent = TextInput(hint_text="EMP-001", multiline=False, size_hint_y=None, height=dp(44), background_color=SURF2, foreground_color=WHITE)
        v.add_widget(ent)

        def confirm(*_):
            code = ent.text.strip().upper()
            if code in EMPLOYEES:
                del EMPLOYEES[code]
                save_employees(EMPLOYEES)
            p.dismiss()

        b = Button(text="Remove", background_color=(0.97,0.36,0.36,1), background_normal="", size_hint_y=None, height=dp(44))
        b.bind(on_release=confirm)
        v.add_widget(b)
        p.content = v
        p.open()

# ─────────────────────────────────────────────
#  CAMERA BARCODE SCANNER POPUP (HIGH CONTRAST & FULL-SCREEN)
# ─────────────────────────────────────────────
class BarcodeScanPopup(Popup):
    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.title        = "📷  Scan Barcode"
        self.size_hint    = (1, 1)
        self.auto_dismiss = False
        self.callback     = callback
        self._cap         = None
        self._event       = None
        self._last_result = None
        self._debounce    = 0

        root = BoxLayout(orientation="vertical", spacing=0, padding=0)

        from kivy.uix.relativelayout import RelativeLayout
        cam_container = RelativeLayout()

        # ตัวแปรแสดงภาพสแกน
        self._cam_img = KivyImage(allow_stretch=True, keep_ratio=False, size_hint=(1, 1))
        cam_container.add_widget(self._cam_img)

        # วาดเส้นเล็งตรงกลางหน้าจอไกด์ผู้ใช้งาน
        from kivy.uix.widget import Widget as _W
        from kivy.graphics  import Line, Color as GColor
        overlay = _W(size_hint=(1, 1))

        def _draw_overlay(widget, *_):
            widget.canvas.clear()
            w, h = widget.size
            cx, cy = widget.center

            with widget.canvas:
                # ทำพื้นหลังขอบบน-ล่างให้ทึบลงเล็กน้อยเพื่อให้เห็นเส้นไกด์ชัดเจน
                GColor(0, 0, 0, 0.3)
                from kivy.graphics import Rectangle
                strip_h = h * 0.28
                Rectangle(pos=(0, h - strip_h), size=(w, strip_h))
                Rectangle(pos=(0, 0),           size=(w, strip_h))

                box_w = w * 0.85
                box_h = h * 0.25
                bx    = cx - box_w / 2
                by    = cy - box_h / 2

                # วาดกรอบสี่เหลี่ยมเป้าหมายกลางจอ
                GColor(0.31, 0.56, 0.97, 1)
                Line(rectangle=(bx, by, box_w, box_h), width=dp(2))

                # เส้นยิงเลเซอร์สีเขียวตรงกลางสแกนเนอร์
                GColor(0.22, 0.85, 0.59, 0.8)
                Line(points=[bx + dp(10), cy, bx + box_w - dp(10), cy], width=dp(2))

        overlay.bind(size=_draw_overlay, pos=_draw_overlay)
        cam_container.add_widget(overlay)

        hint = Label(text="Point camera at [b]Barcode (Code 128)[/b]", markup=True, font_size=dp(14), color=(1, 1, 1, 0.9), size_hint=(1, None), height=dp(36), pos_hint={"top": 0.98}, halign="center")
        hint.bind(size=lambda w, s: setattr(w, "text_size", s))
        cam_container.add_widget(hint)

        self._result_lbl = Label(text="Align barcode with the green line", font_size=dp(15), bold=True, color=(0.91,0.92,0.97,1), size_hint=(1, None), height=dp(38), pos_hint={"y": 0.02}, halign="center")
        self._result_lbl.bind(size=lambda w, s: setattr(w, "text_size", s))
        cam_container.add_widget(self._result_lbl)

        root.add_widget(cam_container)

        # เมนูปุ่มด้านล่าง
        bottom = BoxLayout(size_hint_y=None, height=dp(58), spacing=dp(8), padding=[dp(12), dp(6)])
        with bottom.canvas.before:
            from kivy.graphics import Color as GC, Rectangle as GR
            GC(0.059, 0.067, 0.090, 1)
            self._bot_rect = GR(pos=bottom.pos, size=bottom.size)
        bottom.bind(pos =lambda w, v: setattr(self._bot_rect, "pos",  v), size=lambda w, v: setattr(self._bot_rect, "size", v))

        self._cam_idx_lbl = Label(text="CAM 0", font_size=dp(11), color=(0.48, 0.50, 0.60, 1), size_hint_x=0.25)
        bottom.add_widget(self._cam_idx_lbl)

        switch_btn = Button(text="🔄 Switch Cam", background_color=(0.145, 0.157, 0.212, 1), background_normal="", font_size=dp(12), size_hint_x=0.35)
        switch_btn.bind(on_release=self._switch_cam)
        bottom.add_widget(switch_btn)

        cancel_btn = Button(text="✕ Cancel", background_color=(0.97, 0.36, 0.36, 1), background_normal="", font_size=dp(13), bold=True, size_hint_x=0.40)
        cancel_btn.bind(on_release=self._close)
        bottom.add_widget(cancel_btn)

        root.add_widget(bottom)
        self.content = root

        self._cam_index = 0
        self.bind(on_open=self._start)

    def _start(self, *_):
        self._open_cam(self._cam_index)

    def _open_cam(self, idx):
        if self._cap and self._cap.isOpened(): self._cap.release()
        if self._event: self._event.cancel()

        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            self._result_lbl.text  = f"❌  Camera {idx} not found"
            self._result_lbl.color = (0.97, 0.36, 0.36, 1)
            return

        # ตั้งค่าขนาดความละเอียดของภาพให้คมชัดเพื่อสแกนบาร์โค้ดขนาดเล็กได้ดีขึ้น
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS,       1) # เปิดระบบโฟกัสอัตโนมัติ

        self._cam_index = idx
        self._cam_idx_lbl.text = f"CAM {idx}"
        self._event = Clock.schedule_interval(self._update, 1 / 25)

    def _switch_cam(self, *_):
        self._open_cam(1 - self._cam_index)

    def _close(self, *_):
        if self._event: self._event.cancel()
        if self._cap and self._cap.isOpened(): self._cap.release()
        self.dismiss()

    def _update(self, dt):
        if not self._cap or not self._cap.isOpened(): return
        ret, frame = self._cap.read()
        if not ret: return

        # พลิกซ้ายขวาตามทิศทางกล้องหน้า/หลังจริง
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        decoded_value = None
        if PYZBAR_OK:
            # 💡 [ปรับปรุงส่วนสำคัญ] ทำ Grayscale และยกระดับ Contrast เพิ่มเป็น 1.5 เท่า ขจัดสัญญาณรบกวน
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=15)

            # ถอดรหัสบาร์โค้ดจากแผ่นภาพความคมชัดสูงแบบเต็มหน้าจอ (ป้องกันหลุดขอบครอป)
            barcodes = _pyzbar.decode(gray)
            for bc in barcodes:
                val = bc.data.decode("utf-8", errors="ignore").strip()
                if not val: continue

                # วาดเส้นไฮไลต์รอบตำแหน่งบาร์โค้ดที่ระบบตรวจพบ
                pts = bc.polygon
                if len(pts) >= 4:
                    pts_np = np.array([(p.x, p.y) for p in pts], dtype=np.int32)
                    cv2.polylines(frame, [pts_np], True, (56, 217, 150), 3)

                cv2.putText(frame, val, (bc.rect.left, bc.rect.top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 217, 150), 2)
                decoded_value = val
                break

        # ตรวจสอบว่าแอปสแกนได้ค่าเดิมเสถียร 3 เฟรมซ้อน (Debounce) เพื่อป้องกันการอ่านรหัสผิดพลาด
        if decoded_value:
            if decoded_value == self._last_result:
                self._debounce += 1
            else:
                self._last_result = decoded_value
                self._debounce    = 1

            self._result_lbl.text  = f"🔍  Found: {decoded_value}"
            self._result_lbl.color = (0.97, 0.79, 0.28, 1)

            if self._debounce >= 3:
                self._result_lbl.text  = f"✅  Confirmed: {decoded_value}"
                self._result_lbl.color = (0.22, 0.85, 0.59, 1)
                confirmed = decoded_value
                self._close()
                self.callback(confirmed)
                return
        else:
            if self._debounce > 0: self._debounce -= 1
            else: self._last_result = None

        # แปลงข้อมูลสีกลับมาอัปเดตลงบนจอแสดงผลของ Kivy GUI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 0)
        tex = Texture.create(size=(w, h), colorfmt="rgb")
        tex.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self._cam_img.texture = tex


# ─────────────────────────────────────────────
#  MAIN SCREEN (หน้าจอหลักเชื่อมต่อกล้องสมบูรณ์)
# ─────────────────────────────────────────────
class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self.current_emp  = None
        self.doc_number   = None
        self.active_item  = None

        self.padding  = [dp(12), dp(8), dp(12), dp(8)]
        self.spacing  = dp(8)

        # ── Top Bar ──
        top = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        with top.canvas.before:
            Color(*SURFACE)
            self._top_rect = RoundedRectangle(pos=top.pos, size=top.size, radius=[dp(8)])
        top.bind(pos=self._upd_top, size=self._upd_top)

        top.add_widget(Label(text="🏭 WAREHOUSE SCANNER", font_size=dp(15), bold=True, color=(0.31,0.56,0.97,1)))
        self.clock_lbl = Label(text="", font_size=dp(11), color=(0.48,0.50,0.60,1))
        top.add_widget(self.clock_lbl)
        self.add_widget(top)

        # ── Status Banner ──
        self.status_lbl = Label(text="😃  Scan face to log in", font_size=dp(15), bold=True, color=(0.91,0.92,0.97,1), size_hint_y=None, height=dp(72), halign="center", valign="middle")
        self.status_lbl.bind(size=lambda w,s: setattr(w,"text_size",s))
        with self.status_lbl.canvas.before:
            Color(*SURF2)
            self._s_rect = RoundedRectangle(pos=self.status_lbl.pos, size=self.status_lbl.size, radius=[dp(10)])
        self.status_lbl.bind(pos =lambda w,v: setattr(self._s_rect, "pos",  v), size=lambda w,v: setattr(self._s_rect, "size", v))
        self.add_widget(self.status_lbl)

        # ── Info Row ──
        info = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(6))
        self.emp_lbl = Label(text="👤 —", font_size=dp(13), color=(0.48,0.50,0.60,1), bold=True, halign="left", valign="middle")
        self.emp_lbl.bind(size=lambda w,s: setattr(w,"text_size",s))
        self.doc_lbl = Label(text="📄 —", font_size=dp(13), color=(0.48,0.50,0.60,1), bold=True, halign="right", valign="middle")
        self.doc_lbl.bind(size=lambda w,s: setattr(w,"text_size",s))
        info.add_widget(self.emp_lbl)
        info.add_widget(self.doc_lbl)
        self.add_widget(info)

        # ── Barcode Input Row (มีช่องกรอกพวงปุ่มสแกนด่วน) ──
        self.add_widget(Label(
            text="🔍  Scan Barcode:", font_size=dp(13),
            color=(0.48,0.50,0.60,1), size_hint_y=None, height=dp(26), halign="left"))

        barcode_layout = BoxLayout(size_hint_y=None, height=dp(58), spacing=dp(8))

        self.barcode_in = TextInput(
            hint_text="Scan or type barcode...",
            background_color=SURF2,
            foreground_color=(0.91,0.92,0.97,1),
            cursor_color=(0.31,0.56,0.97,1),
            font_size=dp(20),
            multiline=False,
            size_hint_x=0.7)
        self.barcode_in.bind(on_text_validate=self._on_scan)

        # ปุ่มทางลัดกดเปิดกล้องขึ้นมาสแกนทันที
        self.cam_btn = Button(
            text="📷 Scan",
            background_color=ACCENT,
            background_normal="",
            bold=True,
            size_hint_x=0.3)
        self.cam_btn.bind(on_release=self._open_camera_scanner)

        barcode_layout.add_widget(self.barcode_in)
        barcode_layout.add_widget(self.cam_btn)
        self.add_widget(barcode_layout)

        # ── Qty Row ──
        qty_row = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(10))
        qty_row.add_widget(Label(text="Qty:", font_size=dp(14), bold=True, color=(0.48,0.50,0.60,1), size_hint_x=0.3))
        self.qty_in = TextInput(text="1", background_color=SURF2, foreground_color=(0.97,0.79,0.28,1), cursor_color=(0.97,0.79,0.28,1), font_size=dp(22), multiline=False, input_filter="int", size_hint_x=0.35)
        self.qty_in.bind(on_text_validate=self._confirm)
        qty_row.add_widget(self.qty_in)
        self.unit_lbl = Label(text="unit", font_size=dp(13), color=(0.48,0.50,0.60,1), size_hint_x=0.35)
        qty_row.add_widget(self.unit_lbl)
        self.add_widget(qty_row)

        # ── Confirm Button ──
        conf_btn = Button(text="✅  Confirm & Save", background_color=(0.22,0.85,0.59,1), background_normal="", bold=True, font_size=dp(16), size_hint_y=None, height=dp(56))
        conf_btn.bind(on_release=self._confirm)
        self.add_widget(conf_btn)

        # ── Action Buttons ──
        btn_row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(6))

        def mk(txt, col, cb):
            b = Button(text=txt, background_color=col, background_normal="", font_size=dp(13), bold=True)
            b.bind(on_release=lambda *_: cb())
            btn_row.add_widget(b)

        mk("📸 Face Login", (0.31,0.56,0.97,1), self._face_login)
        mk("😴 Logout",    (0.97,0.36,0.36,1), self._logout)
        mk("⚙️ Staff",     (0.18,0.19,0.26,1), self._open_manager)
        self.add_widget(btn_row)

        # ── Guide label ──
        self.guide_lbl = Label(text="Steps: Face ➔ Doc ➔ Header ➔ Product ➔ Qty", font_size=dp(11), color=(0.48,0.50,0.60,1), size_hint_y=None, height=dp(30), halign="center")
        self.guide_lbl.bind(size=lambda w,s: setattr(w,"text_size",s))
        self.add_widget(self.guide_lbl)

        self.add_widget(Widget())
        Clock.schedule_interval(self._tick, 1)

    def _upd_top(self, w, *_):
        self._top_rect.pos  = w.pos
        self._top_rect.size = w.size

    def _tick(self, dt):
        now     = datetime.datetime.now()
        year_th = now.year + 543
        mo      = MONTH_THAI[now.month]
        self.clock_lbl.text = now.strftime(f"%d {mo} {year_th} %H:%M:%S")

    # ── Camera Scanner Callbacks ──
    def _open_camera_scanner(self, *_):
        if not self.current_emp:
            self.status_lbl.text  = "😠  Not logged in! Tap Face Login first"
            self.status_lbl.color = (0.97,0.36,0.36,1)
            return
        BarcodeScanPopup(callback=self._on_scan_from_cam).open()

    def _on_scan_from_cam(self, barcode_data):
        self.barcode_in.text = barcode_data
        self._on_scan()

    # ── Face login ──────────────────────────
    def _face_login(self):
        FaceLoginPopup(callback=self._login_cb).open()

    def _login_cb(self, code):
        if code in EMPLOYEES:
            self.current_emp = EMPLOYEES[code]
            self.emp_lbl.text  = f"👤 {self.current_emp}"
            self.emp_lbl.color = (0.22,0.85,0.59,1)
            self.status_lbl.text  = "✅  Logged in — scan document number"
            self.status_lbl.color = (0.22,0.85,0.59,1)

    def _logout(self):
        self.current_emp = None
        self.doc_number  = None
        self.active_item = None
        self.emp_lbl.text   = "👤 —"
        self.emp_lbl.color  = (0.48,0.50,0.60,1)
        self.doc_lbl.text   = "📄 —"
        self.doc_lbl.color  = (0.48,0.50,0.60,1)
        self.status_lbl.text  = "😃  Scan face to log in"
        self.status_lbl.color = (0.91,0.92,0.97,1)
        self.barcode_in.text  = ""

    def _open_manager(self):
        EmployeeManagerPopup().open()

    # ── Scan logic ──────────────────────────
    def _on_scan(self, *_):
        data = self.barcode_in.text.upper().strip()
        self.barcode_in.text = ""
        if not data: return

        if data == "LOGOUT":
            self._logout(); return

        if data == "EXIT":
            if self.doc_number:
                self.doc_number  = None
                self.active_item = None
                self.doc_lbl.text  = "📄 —"
                self.doc_lbl.color = (0.48,0.50,0.60,1)
                self.status_lbl.text  = "🔄  Cleared — scan new document"
                self.status_lbl.color = (0.97,0.79,0.28,1)
            return

        if not self.current_emp:
            self.status_lbl.text  = "😠  Not logged in! Tap Face Login first"
            self.status_lbl.color = (0.97,0.36,0.36,1)
            return

        if data not in MODE_BARCODES and data not in PRODUCTS:
            if not self.doc_number:
                self.doc_number    = data
                self.doc_lbl.text  = f"📄 {data}"
                self.doc_lbl.color = (0.97,0.79,0.28,1)
                self.status_lbl.text  = f"✅  Doc #{data} — scan product header"
                self.status_lbl.color = (0.97,0.79,0.28,1)
            else:
                self.status_lbl.text  = f"❌  Already have doc #{self.doc_number}"
                self.status_lbl.color = (0.97,0.36,0.36,1)
            return

        if not self.doc_number:
            self.status_lbl.text  = "❌  No document! Scan doc first"
            self.status_lbl.color = (0.97,0.36,0.36,1)
            return

        if data in MODE_BARCODES:
            self.active_item       = MODE_BARCODES[data]
            prod                   = PRODUCTS[self.active_item]
            self.unit_lbl.text     = prod["unit"]
            self.status_lbl.text   = f"🏷️  {prod['name']} — scan item barcode"
            self.status_lbl.color  = (0.31,0.56,0.97,1)
            return

        if data == self.active_item:
            self.qty_in.focus = True
            self.status_lbl.text  = "✅  Product OK — enter quantity"
            self.status_lbl.color = (0.22,0.85,0.59,1)
        else:
            self.status_lbl.text  = f"❌  Wrong product! Expected {self.active_item}"
            self.status_lbl.color = (0.97,0.36,0.36,1)

    def _confirm(self, *_):
        if not all([self.current_emp, self.doc_number, self.active_item]):
            self.status_lbl.text  = "❌  Complete all steps first"
            self.status_lbl.color = (0.97,0.36,0.36,1)
            return
        try:
            qty = int(self.qty_in.text)
            if qty <= 0: raise ValueError
        except ValueError:
            self.status_lbl.text  = "❌  Invalid quantity"
            self.status_lbl.color = (0.97,0.36,0.36,1)
            return

        doc, emp, item = self.doc_number, self.current_emp, self.active_item
        threading.Thread(target=save_transaction, args=(doc, emp, item, qty), daemon=True).start()

        prod = PRODUCTS[item]
        self.status_lbl.text  = f"💾  Saved {prod['name']} ×{qty} — scan next"
        self.status_lbl.color = (0.22,0.85,0.59,1)
        self.active_item      = None
        self.qty_in.text      = "1"
        self.barcode_in.focus = True

# ─────────────────────────────────────────────
#  KIVY APP ENTRYPOINT
# ─────────────────────────────────────────────
class WarehouseApp(App):
    def build(self):
        Builder.load_string(KV)
        if not init_models():
            p = Popup(title="Model Not Found", content=Label(text="Download face_detection_yunet_2023mar.onnx\nand place it in the app folder.", halign="center"), size_hint=(.85,.4))
            p.open()

        self.title = "Warehouse Scanner"

        root = BoxLayout(orientation="vertical")
        with root.canvas.before:
            Color(0.059, 0.067, 0.090, 1)
            self._bg = RoundedRectangle(pos=root.pos, size=root.size)
        root.bind(pos =lambda w,v: setattr(self._bg,"pos",v), size=lambda w,v: setattr(self._bg,"size",v))

        root.add_widget(MainScreen())
        return root

if __name__ == "__main__":
    WarehouseApp().run()
