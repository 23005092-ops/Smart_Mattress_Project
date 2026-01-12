import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
from datetime import datetime
import json
import os
import uuid
import math
import streamlit.components.v1 as components  # used for small client-side DOM cleanup

# --- 1. SYSTEM CONFIGURATION (Easy Switch for Later) ---
# When you have the Pi connected to Arduino, change this to TRUE
REAL_SENSORS_CONNECTED = False 
SERIAL_PORT = '/dev/ttyUSB0' # Standard for Raspberry Pi
BAUD_RATE = 9600
# Sleep history persistence paths
SLEEP_HISTORY_CSV_PATH = os.path.join(os.path.dirname(__file__), 'sleep_history.csv') if '__file__' in globals() else 'sleep_history.csv'
SLEEP_HISTORY_JSON_PATH = os.path.join(os.path.dirname(__file__), 'sleep_history.json') if '__file__' in globals() else 'sleep_history.json'

# --- Audit log (global helpers) ---
AUDIT_LOG_PATH = os.path.join(os.path.dirname(__file__), 'audit_log.json') if '__file__' in globals() else 'audit_log.json'

# Path to persist learned bladder policy/model (disabled)

def append_audit(action, user='unknown', target_id=None, meta=None):
    """Append an audit entry to the global audit log (safe to call anywhere)."""
    entry = {'timestamp': datetime.now().isoformat(), 'action': action, 'user': user, 'target_id': target_id, 'meta': meta}
    logs = []
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    try:
        with open(AUDIT_LOG_PATH, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    except Exception:
        pass


def load_audit_logs():
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Auto-load of persisted bladder policy disabled (bladder policy tools removed)

# --- 2. PAGE SETUP ---
st.set_page_config(
    page_title="Smart Mattress AI Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. AESTHETIC STYLING (CSS) ---
st.markdown("""
<style>
    /* 1. Set Main Background to Dark (Streamlit Default is fine, but we ensure consistency) */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    
    /* 2. STYLE THE METRIC BOXES (Darker Grey + Borders) */
    /* Target multiple metric container variations (main area + sidebar) */
    div[data-testid="stMetric"],
    div[data-testid="metric-container"],
    section[data-testid="stMetric"],
    .stMetric {
        background-color: #2b2b2b !important; /* Dark Grey */
        border: 1px solid #41444d;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.3);
    }

    /* Ensure sidebar metric containers also pick up the darker background */
    [data-testid="stSidebar"] div[data-testid="stMetric"],
    .stSidebar div[data-testid="stMetric"],
    .sidebar div[data-testid="stMetric"] {
        background-color: #2b2b2b !important;
    }
    
    /* 3. CHANGE FONT for the Label (e.g. "Peak Pressure") */
    div[data-testid="stMetricLabel"] p {
        color: #babcbf !important; /* Light Grey Text */
        font-size: 14px !important;
        font-family: 'Source Sans Pro', sans-serif;
    }

    /* 4. CHANGE FONT for the Value (e.g. "25.4 mmHg") - SMALLER & NICER */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important; /* White Text */
        font-size: 26px !important; /* Smaller than default */
        font-family: 'Segoe UI', Roboto, sans-serif; /* Aesthetic Font */
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. DATA ENGINE (Simulated vs Real) ---
def get_pressure_data(mode='Anti-bedsore', demo=False, massage_intensity=0.5):
    """Return a grid and sensors_df according to operation mode.
    - mode: 'Anti-bedsore' -> real-use behavior; in our environment this will indicate 'no human on bed' (low pressures)
    - mode: 'Massage Mode' -> produce a sinusoidal cradle/massage waveform influenced by massage_intensity
    - demo: when True, always produce simulated noisy data (useful for demos)

    The function returns (grid, sensors_df)
    """
    rng = np.random.default_rng()
    t = time.time()

    # sensor labels (7 sensors)
    sensor_labels = [
        "S1 - Left Shoulder",
        "S2 - Right Shoulder",
        "S3 - Lower Back",
        "S4 - Left Hand Area",
        "S5 - Right Hand Area",
        "S6 - Left Leg",
        "S7 - Right Leg",
    ]

    if demo:
        # Demo: same behavior as previous 'Simulation' - noisy with occasional spikes
        vals = rng.normal(20, 2, size=(7,))
        if rng.random() > 0.6:
            spike_idx = rng.choice(7)
            vals[spike_idx] += rng.uniform(10, 30)
    else:
        # Non-demo: mode-specific behaviors
        if mode == 'Anti-bedsore':
            # Real-use (no human on bed) -> very low baseline static values with small noise
            baseline = np.zeros(7)
            vals = baseline + rng.normal(0, 0.2, size=7)
        elif mode == 'Massage Mode':
            # Produce sinusoidal waveform across sensors to emulate cradle/massage
            base = np.array([5.0,5.0,4.0,3.0,3.5,2.0,2.5]) * massage_intensity
            wave = np.sin(np.linspace(0, 2*math.pi, 7) + (t/3.0)) * (8.0 * massage_intensity)
            vals = base + wave + rng.normal(0, 0.4, size=7)
        else:
            # fallback: small noise
            vals = rng.normal(0, 0.2, size=7)

    sensors_df = pd.DataFrame({
        "sensor_id": [f"S{i+1}" for i in range(7)],
        "location": sensor_labels,
        "mmHg": np.round(vals, 1)
    })

    # Map sensors to a 3x3 grid for visualization
    grid = pd.DataFrame(np.nan, index=["Shoulders", "Torso", "Legs"], columns=["Left", "Center", "Right"])
    v = vals
    grid.at['Shoulders','Left'] = v[0]
    grid.at['Shoulders','Right'] = v[1]
    grid.at['Torso','Left'] = v[3]
    grid.at['Torso','Center'] = v[2]
    grid.at['Torso','Right'] = v[4]
    grid.at['Legs','Left'] = v[5]
    grid.at['Legs','Right'] = v[6]
    grid = grid.round(1)

    return grid, sensors_df

# --- 5a. PREDICTION HOOK (PLACEHOLDER for AI MODEL) ---
def predict_time_to_stage1(sensors_df, threshold=35.0, base_hours=72.0):
    """Estimate time (hours) until Stage I if patient is not moved.

    Current simple heuristic: at pressure == threshold, expected time == base_hours.
    For pressure > threshold the time reduces roughly proportional to (threshold / pressure).

    Replace this function with your ML model later. The model should accept the
    `sensors_df` and optionally `history` and return estimated hours until Stage I.
    """
    current_pressure = float(sensors_df['mmHg'].max())
    if current_pressure <= threshold:
        return float('inf')
    est_hours = base_hours * (threshold / current_pressure)
    return max(0.1, est_hours)

# --- 5b. EXPOSURE ACCUMULATION (simple cumulative load model) ---
def accumulate_exposure(sensors_df, delta_minutes=1.0, threshold=30.0):
    """Accumulate exposure in mmHg*minutes per sensor.

    For each sensor: exposure += max(0, pressure - threshold) * delta_minutes
    Returns total exposure (sum of all per-sensor mmHg*minutes) and the updated per-sensor exposures dict.
    """
    if 'sensor_exposure' not in st.session_state:
        # initialize exposures to zero
        st.session_state['sensor_exposure'] = {sid: 0.0 for sid in sensors_df['sensor_id']}

    exposures = st.session_state['sensor_exposure']
    total = 0.0
    for _, row in sensors_df.iterrows():
        sid = row['sensor_id']
        pressure = float(row['mmHg'])
        excess = max(0.0, pressure - threshold)
        add = excess * delta_minutes
        exposures[sid] = exposures.get(sid, 0.0) + add
        total += exposures[sid]

    st.session_state['sensor_exposure'] = exposures
    return total, exposures

# --- 5c. PREDICTION FROM EXPOSURE THRESHOLD ---
def predict_time_to_stage1_from_exposure(total_exposure, per_minute_excess, exposure_threshold=5000.0):
    """Predict hours until Stage I based on cumulative exposure and current per-minute excess.

    - total_exposure: current cumulative mmHg*minutes
    - per_minute_excess: current rate (mmHg per minute) summed across sensors
    - exposure_threshold: mmHg*minutes value that maps to Stage I (tunable)
    """
    if per_minute_excess <= 0:
        return float('inf')
    remaining_minutes = max(0.0, (exposure_threshold - total_exposure) / per_minute_excess)
    return remaining_minutes / 60.0

# --- 5d. ML MODEL HELPERS (demo linear model + JSON load/save) ---

# --- Helper: disease risk computations ---

def compute_risks(rec):
    """Compute heuristic risk percentages for common conditions based on sleep metrics."""
    risks = {}
    eff = rec.get('sleep_eff', 80)
    hr = rec.get('avg_hr', 60)
    spo2 = rec.get('avg_spo2', 97)
    total_exposure = rec.get('total_exposure', 0.0)

    # Sleep apnea risk
    apnea = 0.0
    if spo2 < 94:
        apnea += (94 - spo2) * 4.0
    apnea += max(0.0, (75 - eff) * 0.5)
    apnea = np.clip(apnea, 0, 100)
    risks['Sleep Apnea'] = float(apnea)

    # Cardiovascular risk
    cardio = max(0.0, (hr - 60) * 1.2) + max(0.0, (80 - eff) * 0.6) + (float(total_exposure) / 1000.0)
    cardio = np.clip(cardio, 0, 100)
    risks['Cardiovascular'] = float(cardio)

    # Diabetes-ish risk (sleep disturbance related)
    diabetes = max(0.0, (80 - eff) * 0.8) + max(0.0, (6 - rec.get('total_sleep_h', 7)) * 6.0)
    diabetes = np.clip(diabetes, 0, 100)
    risks['Metabolic / Diabetes'] = float(diabetes)

    # Hypertension risk (related to high HR and low sleep)
    hyper = max(0.0, (hr - 65) * 1.5) + max(0.0, (80 - eff) * 0.4)
    hyper = np.clip(hyper, 0, 100)
    risks['Hypertension'] = float(hyper)

    # Alzheimer's-like risk heuristic (related to low REM/deep sleep and poor efficiency)
    rem = rec.get('rem_pct', 18)
    deep = rec.get('deep_pct', 18)
    alz = max(0.0, (70 - eff) * 0.6) + max(0.0, (20 - rem) * 1.2) + max(0.0, (20 - deep) * 1.0)
    alz = np.clip(alz, 0, 100)
    risks['Alzheimer-like'] = float(alz)
    return risks

def build_features(sensors_df, total_exposure, per_minute_excess):
    """Return feature vector in order matching the demo model."""
    max_p = float(sensors_df['mmHg'].max())
    avg_p = float(sensors_df['mmHg'].mean())
    return np.array([max_p, avg_p, total_exposure, per_minute_excess], dtype=float)


def train_demo_linear_model(n_samples=800, threshold_pressure=30.0, exposure_threshold=5000.0, base_hours=72.0):
    """Train a simple linear model from synthetic data and return a dict model.

    Model format: {'intercept': float, 'coef': [c1, c2, ...], 'features': [..]}
    """
    rng = np.random.default_rng(seed=42)
    X = []
    y = []
    for _ in range(n_samples):
        max_p = rng.normal(20, 6)
        avg_p = max_p - rng.normal(0, 2)
        total_exposure = rng.uniform(0, exposure_threshold * 1.5)
        per_minute_excess = rng.uniform(0, max(0.0, max_p - threshold_pressure))

        # Generate a target hours value using a simple exposure model + noise
        if per_minute_excess <= 0:
            hours = base_hours * (1.0 + (exposure_threshold - total_exposure) / exposure_threshold)
        else:
            remaining_minutes = max(0.0, (exposure_threshold - total_exposure) / per_minute_excess)
            hours = remaining_minutes / 60.0

        hours = hours + rng.normal(0, max(0.1, hours * 0.05))
        X.append([max(0, max_p), max(0, avg_p), total_exposure, per_minute_excess])
        y.append(hours)

    X = np.array(X)
    y = np.array(y)
    A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coef[0])
    coefs = [float(c) for c in coef[1:]]
    model = {'intercept': intercept, 'coef': coefs, 'features': ['max_pressure','avg_pressure','total_exposure','per_minute_excess']}
    return model


def predict_with_model(model, sensors_df, total_exposure, per_minute_excess):
    feats = build_features(sensors_df, total_exposure, per_minute_excess)
    pred = model['intercept'] + float(np.dot(model['coef'], feats))
    return max(0.0, float(pred))


def save_model_json(model, path):
    with open(path, 'w') as f:
        json.dump(model, f)


def load_model_json_filelike(file_like):
    try:
        data = json.load(file_like)
        if not all(k in data for k in ('intercept', 'coef', 'features')):
            raise ValueError('JSON missing required keys (intercept, coef, features)')
        return data
    except Exception as e:
        raise

# --- Sleep detection helpers ---
def _finalize_sleep_record(start_ts, end_ts, sensors_df=None, vitals=None):
    """Build a daily record based on start/end timestamps and lightweight summaries."""
    duration_minutes = max(0, (end_ts - start_ts).total_seconds() / 60.0)
    hours = round(duration_minutes / 60.0, 2)
    # Build simple interval qualities (30-min bins)
    n_intervals = max(1, int(math.ceil(hours * 2)))
    base_eff = int(np.clip(78 - max(0, 8 - hours) * 2 + np.random.normal(0,6), 30, 100))
    intervals = []
    for i in range(n_intervals):
        q = int(np.clip(base_eff + np.random.normal(0,8), 20, 100))
        intervals.append({'index': i, 'quality': int(q)})
    rec = {
        'id': str(uuid.uuid4()),
        'timestamp': start_ts.date().isoformat() + 'T07:00:00',
        'sleep_score': int(np.clip(base_eff + np.random.normal(0,4), 30, 100)),
        'total_sleep_h': hours,
        'sleep_eff': base_eff,
        'rem_pct': int(np.clip(18 + np.random.normal(0,6), 0, 60)),
        'deep_pct': int(np.clip(18 + np.random.normal(0,6), 0, 60)),
        'light_pct': max(0, 100 - (int(np.clip(18 + np.random.normal(0,6), 0, 60)) + int(np.clip(18 + np.random.normal(0,6), 0, 60)))),
        'avg_hr': round(float(vitals['hr']) if vitals else round(float(np.clip(62 + np.random.normal(0,3), 30, 110)),1),1),
        'avg_rr': round(float(vitals['rr']) if vitals else 14.0,1),
        'avg_spo2': round(float(vitals['spo2']) if vitals else float(np.clip(97 + np.random.normal(0,0.8), 85, 100)),1),
        'intervals': intervals,
        'auto': False,
        'locked': False,
        'deleted': False,
    }
    # Append and persist
    st.session_state['sleep_history'].append(rec)
    # Use session-backed persist flag (defaults to True) so this works even when called from other pages
    if st.session_state.get('persist_sleep', True):
        save_sleep_history(st.session_state['sleep_history'])
    append_audit('auto_recorded_by_detector', user='system', target_id=rec['id'], meta={'start':str(start_ts), 'end':str(end_ts), 'total_h':hours})

    return rec


def update_sleep_state(sensors_df, avg_pressure, vitals):
    """Simple hysteresis-based sleep detector. Uses sensor pressure + stability to detect sleep start/end.
    - If average pressure > threshold_pressure * 0.6 for several checks -> presence
    - If presence persists and variance low for N checks -> mark as sleeping
    - When sleeping and pressure drops -> mark wake and finalize record
    """
    # initialize state
    if 'sleep_detector' not in st.session_state:
        st.session_state['sleep_detector'] = {'state':'AWAKE', 'presence_counter':0, 'stable_counter':0, 'start_ts':None}
    s = st.session_state['sleep_detector']

    presence_threshold = max(8.0, float(threshold_pressure) * 0.6)
    current_present = float(sensors_df['mmHg'].max()) > presence_threshold

    # update presence counter
    if current_present:
        s['presence_counter'] = min(10, s['presence_counter'] + 1)
    else:
        s['presence_counter'] = max(0, s['presence_counter'] - 1)

    # Check for stable low-variation indicating sleep (low movement)
    recent = np.array(st.session_state.get('pressure_history', [])[-12:])
    var = float(np.nanvar(recent)) if recent.size>0 else 0.0
    stable = var < 1.5
    if stable and s['presence_counter'] >= 2:
        s['stable_counter'] = min(10, s['stable_counter'] + 1)
    else:
        s['stable_counter'] = max(0, s['stable_counter'] - 1)

    # State transitions
    if s['state'] == 'AWAKE' and s['presence_counter'] >= 3 and s['stable_counter'] >= 3:
        # start sleeping
        s['state'] = 'SLEEPING'
        s['start_ts'] = datetime.now()
        append_audit('sleep_started', user='system', target_id=None, meta={'start': str(s['start_ts'])})
    elif s['state'] == 'SLEEPING' and (s['presence_counter'] == 0 or s['stable_counter'] == 0 and not current_present):
        # woke up
        end_ts = datetime.now()
        start_ts = s.get('start_ts', end_ts)
        if (end_ts - start_ts).total_seconds() > 60*30:  # only record if at least 30 minutes
            rec = _finalize_sleep_record(start_ts, end_ts, sensors_df=sensors_df, vitals=vitals)
            st.info(f"Recorded sleep session ({rec['total_sleep_h']:.2f} h)")
        append_audit('sleep_ended', user='system', target_id=None, meta={'start': str(start_ts), 'end': str(end_ts)})
        s['state'] = 'AWAKE'
        s['start_ts'] = None

    st.session_state['sleep_detector'] = s

# --- 5. SIDEBAR (System Health) ---
st.sidebar.title("System Status")

# Device Mode and Page Selector
device_mode = st.sidebar.selectbox("Device Mode", ["Anti-bedsore", "Massage Mode"], index=0, help="Choose operation mode: Anti-bedsore or Massage Mode.")
# Clear transient UI-render flags each run to avoid duplicate widgets appearing after emergency actions
page = st.sidebar.radio("Page", ["Dashboard", "Sleep Tracking"]) 
# Initialize render guards (persist across reruns to avoid accidental double-rendering)
if 'hr_metric_rendered' not in st.session_state:
    st.session_state['hr_metric_rendered'] = False
# Track last page to reset page-local render flags when user navigates
if 'last_page' not in st.session_state:
    st.session_state['last_page'] = page
elif st.session_state['last_page'] != page:
    st.session_state['last_page'] = page
    st.session_state['hr_metric_rendered'] = False
    # Clear a few transient page-specific flags to avoid leaking Sleep Tracking UI into the Dashboard
    if page == 'Dashboard':
        st.session_state.pop('doctor_view', None)
        st.session_state.pop('doctor_demo_generated', None)
        # Remove stale widget keys created by Sleep Tracking
        st.session_state.pop('select_sleep_record', None)
        st.session_state.pop('close_doctor_view', None)

# Create a persistent placeholder we can empty when navigating away from Sleep Tracking
sleep_placeholder = st.empty()

# Demo override (global): big visible toggle for demoing with simulated data
if 'demo_override' not in st.session_state:
    st.session_state['demo_override'] = False
if not st.session_state['demo_override']:
    if st.sidebar.button('▶ Run Demo (use simulated sensor data)', key='start_demo'):
        st.session_state['demo_override'] = True
        st.sidebar.success('Demo mode enabled (simulated data)')
else:
    if st.sidebar.button('■ Stop Demo (return to real data)', key='stop_demo'):
        st.session_state['demo_override'] = False
        st.sidebar.info('Demo mode disabled (real data resumes)')

# Massage intensity (default)
massage_intensity = 0.5
if device_mode == 'Massage Mode':
    massage_intensity = st.sidebar.slider('Massage intensity', min_value=0.0, max_value=1.0, value=0.5, step=0.05, help='Higher values produce stronger cradle/actuation effects')

mode_label = device_mode + (' (DEMO)' if st.session_state.get('demo_override', False) else '')
st.sidebar.caption(f"Device ID: UM-IOT-001 | Mode: {mode_label}")

# Compact Metrics in Sidebar
c1, c2 = st.sidebar.columns(2)
c1.metric("Battery", "94%", "Charging")
c2.metric("WiFi", "52ms", "Stable")

st.sidebar.divider()
st.sidebar.subheader("System Settings")

# Exposure / Prediction Tuning
st.sidebar.divider()
st.sidebar.subheader("Exposure / Prediction Tuning")
threshold_pressure = st.sidebar.slider("Pressure Threshold (mmHg)", min_value=10, max_value=60, value=35, step=1)
exposure_threshold = st.sidebar.slider("Exposure Threshold (mmHg·min) for Stage I", min_value=500, max_value=20000, value=5000, step=100)
base_hours = st.sidebar.slider("Base Hours at threshold (hours)", min_value=12, max_value=168, value=72, step=1)
# Bladder speed control
max_bladder_speed = st.sidebar.slider("Bladder max change (cm/sec)", min_value=0.05, max_value=2.0, value=0.3, step=0.05)
# Bladder targets are automatic and cannot be manually set
st.sidebar.caption("Adjust thresholds to match clinical ground truth. Changes apply immediately. Bladder targeting is automatic; manual targeting is disabled.")

# Prediction Model Controls
st.sidebar.divider()
st.sidebar.subheader("Prediction Model")
model_choice = st.sidebar.selectbox("Model", ["Heuristic", "Demo Linear Model", "Upload JSON Model"]) 

if model_choice == "Demo Linear Model":
    if st.sidebar.button("Train Demo Model"):
        demo_model = train_demo_linear_model(n_samples=1000, threshold_pressure=threshold_pressure, exposure_threshold=exposure_threshold, base_hours=base_hours)
        st.session_state['ml_model'] = demo_model
        st.sidebar.success("Demo model trained and loaded.")
    if 'ml_model' in st.session_state:
        st.sidebar.write("Loaded model coefficients:")
        st.sidebar.json(st.session_state['ml_model'])
    if 'ml_model' in st.session_state:
        js = json.dumps(st.session_state['ml_model'])
        st.sidebar.download_button("Download model JSON", data=js, file_name="demo_model.json", mime="application/json")

elif model_choice == "Upload JSON Model":
    uploaded = st.sidebar.file_uploader("Upload model JSON", type=["json"]) 
    if uploaded is not None:
        try:
            model = load_model_json_filelike(uploaded)
            st.session_state['ml_model'] = model
            st.sidebar.success("Model loaded from JSON")
        except Exception as e:
            st.sidebar.error(f"Failed to load JSON model: {e}")

# Emergency deflate / reinflate (preserve prev targets to restore on reinflate)
if 'emergency_deflated' not in st.session_state:
    st.session_state['emergency_deflated'] = False

if not st.session_state['emergency_deflated']:
    if st.sidebar.button("🚨 EMERGENCY DEFLATE", type="primary", use_container_width=True, key='em_deflate'):
        with st.spinner("Actuating Solenoid Valves..."):
            time.sleep(1.0)
        # Save previous targets so we can restore them after reinflate
        st.session_state['bladder_targets_prev'] = st.session_state.get('bladder_targets', [5.0,5.0,5.0,5.0]).copy()
        # Set targets to zero immediately (safety), but do not zero the current heights — animate deflation over 10 seconds
        st.session_state['bladder_targets'] = [0.0,0.0,0.0,0.0]
        # record pre-emergency heights and timestamp for smooth 10-second deflation animation
        st.session_state['bladder_current_pre_emergency'] = st.session_state.get('bladder_current', st.session_state.get('bladder_targets_prev', [5.0,5.0,5.0,5.0])).copy()
        st.session_state['emergency_deflate_started_at'] = time.time()
        st.session_state['emergency_deflated'] = True
        try:
            append_audit('emergency_deflated', user='user', target_id=None, meta={'reason':'manual'})
        except Exception:
            pass
        # Intentionally no visible success/banner to avoid changing layout; metrics only are updated.
else:
    if st.sidebar.button("🔼 Reinflate / Resume Actuation", type="primary", use_container_width=True, key='em_reinflate'):
        with st.spinner("Restoring Bladders..."):
            time.sleep(1.0)
        # restore previous targets if present, otherwise use sensible defaults
        if 'bladder_targets_prev' in st.session_state:
            st.session_state['bladder_targets'] = st.session_state.pop('bladder_targets_prev')
        else:
            st.session_state['bladder_targets'] = [5.0, 5.0, 5.0, 5.0]
        # allow smoothing logic to reinflate toward targets
        st.session_state['emergency_deflated'] = False
        # remove deflation animation state so smoothing resumes normal behavior
        st.session_state.pop('emergency_deflate_started_at', None)
        st.session_state.pop('bladder_current_pre_emergency', None)
        try:
            append_audit('emergency_reinflated', user='user', target_id=None, meta=None)
        except Exception:
            pass
        # Intentionally no visible success/banner to avoid changing layout; metrics only are updated.

# --- 6. MAIN DASHBOARD LAYOUT ---
st.title("AI POWERED SMART MATTRESS")

# Client-side cleanup when on Dashboard: hide any leftover Sleep Tracking DOM nodes that
# can occasionally persist due to client-side rendering behaviors (labels like "Sleep Score",
# "Total Sleep", "Sleep Coach Chat", etc.). This runs only when the Dashboard is active.
if page == 'Dashboard':
    # Ensure any server-side placeholder content for Sleep Tracking is removed
    try:
        sleep_placeholder.empty()
    except Exception:
        pass

    components.html('''
    <script>
    (function(){
      const labels = ['sleep score','total sleep','sleep efficiency','avg hr','stage breakdown','sleep coach chat','coach tips','rem','deep','light'];
      function hideLabels(){
        try{
          // Try to remove specific containers by id first (reliable)
          const s = document.getElementById('sleep_coach_container'); if(s) s.remove();
          const p = document.getElementById('sleep_tracking_page'); if(p) p.remove();

          document.querySelectorAll('body *').forEach(el=>{
            // skip if element already cleaned
            if(!el || (el.dataset && el.dataset.cleaned_by_cleanup)) return;
            const text = (el.innerText || '').toLowerCase().trim();
            if(!text) return;
            for(const lab of labels){
              if(text.indexOf(lab) !== -1){
                // hide a reasonable ancestor container (walk up a few levels)
                let node = el;
                for(let i=0;i<10 && node;i++) node = node.parentElement;
                if(node){ node.style.display='none'; node.setAttribute('data-cleaned-by-cleanup','1'); }
                break;
              }
            }
          });
        }catch(e){ /* ignore */ }
      }
      setTimeout(hideLabels, 50);
      let runs=0; let iv=setInterval(()=>{ hideLabels(); if(++runs>40) clearInterval(iv); }, 200);
    })();
    </script>
    ''', height=0)

# Emergency banner intentionally omitted to avoid layout churn when safety actions occur.
# Dashboard will remain visually unchanged; only internal metrics and bladder states update on emergency deflate/reinflate.

# --- Sleep Tracking page (separate view) ---
if page == 'Sleep Tracking':
    # Render all Sleep Tracking UI inside a stable placeholder so we can clear it on page switch
    with sleep_placeholder.container():
        st.header("Sleep Tracking")

        # Persistence toggle (sidebar control also available)
        # Create a checkbox bound to the key 'persist_sleep' so Streamlit handles session state for us
        persist_sleep = st.sidebar.checkbox("Persist sleep history to disk", value=True, key='persist_sleep')

        # Debug marker: visible box to confirm rendering (avoid using the word 'sleep' to bypass cleanup regex)
        st.markdown("<div id='st_rendered_marker' style='border:2px solid #9cff8a;padding:8px;border-radius:6px;color:#9cff8a;margin-bottom:8px;'>ST_RENDERED_MARKER</div>", unsafe_allow_html=True)

        # Reverse any prior client-side cleanup (unhide nodes and remove cleanup flags) so Sleep Tracking shows correctly
        components.html('''
        <script>
        (function(){
          try{
            // Remove the cleanup attribute set earlier and unhide those nodes
            document.querySelectorAll('[data-cleaned-by-cleanup]').forEach(function(e){ e.removeAttribute('data-cleaned-by-cleanup'); e.style.display=''; });
            // Also unhide any known containers by id
            ['sleep_coach_container','sleep_tracking_page','st_rendered_marker'].forEach(function(id){ var el=document.getElementById(id); if(el){ el.style.display=''; }});
            // Unhide any ancestor that contains sleep/coach/deep/light text
            document.querySelectorAll('body *').forEach(function(el){
              try{
                var t = (el.innerText||'').toLowerCase();
                if(t.indexOf('sleep')!==-1 || t.indexOf('coach')!==-1 || t.indexOf('deep')!==-1 || t.indexOf('light')!==-1){
                  var node = el; for(var i=0;i<8 && node;i++){ node.style.display=''; node = node.parentElement; }
                }
              }catch(e){}
            });
          }catch(e){}
        })();
        </script>
        ''', height=0)

        # Debug: show key session state values to help diagnose blank page issues
        st.caption(f"DEBUG: page={page} demo_override={st.session_state.get('demo_override',False)} doctor_view={st.session_state.get('doctor_view',False)} sleep_history_len={len(st.session_state.get('sleep_history',[]))}")

        # Helper: load/save JSON sleep history (keeps meta fields, intervals, deleted flags)
        def load_sleep_history():
            if os.path.exists(SLEEP_HISTORY_JSON_PATH):
                try:
                    with open(SLEEP_HISTORY_JSON_PATH, 'r') as f:
                        return json.load(f)
                except Exception:
                    return []
            # fallback: if CSV exists, load and convert
            if os.path.exists(SLEEP_HISTORY_CSV_PATH):
                try:
                    df = pd.read_csv(SLEEP_HISTORY_CSV_PATH)
                    return df.to_dict(orient='records')
                except Exception:
                    return []
            return []

    def save_sleep_history(list_of_records):
        try:
            with open(SLEEP_HISTORY_JSON_PATH, 'w') as f:
                json.dump(list_of_records, f, indent=2, default=str)
            return True
        except Exception as e:
            st.error(f"Failed to save sleep history: {e}")
            return False

    # Audit helpers are defined globally at the top of the file (use append_audit and load_audit_logs)
    # (Duplicate local definitions removed to ensure global availability.)

    # --- Auto-nightly snapshot helper ---
    def auto_record_if_missing():
        now = datetime.now()
        yesterday = (now - pd.Timedelta(days=1)).date().isoformat()
        # Only run in morning window or when explicitly enabled
        is_morning = 4 <= now.hour <= 12
        if persist_sleep and (is_morning or st.sidebar.checkbox('Force morning check', value=False, key='force_morning_check')):
            # if no record for yesterday, create
            def has_record_for_date(date_str):
                for r in st.session_state['sleep_history']:
                    if r.get('timestamp','').startswith(date_str) and not r.get('deleted', False):
                        return True
                return False
            if not has_record_for_date(yesterday):
                n_intervals = 16  # assume 8 hours -> 16 half-hours as baseline
                intervals = []
                base_eff = int(np.clip(78 + np.random.normal(0,6), 40, 100))
                for i in range(n_intervals):
                    q = int(np.clip(base_eff + np.random.normal(0,8), 20, 100))
                    intervals.append({'index': i, 'quality': int(q)})
                rec = {
                    'id': str(uuid.uuid4()),
                    'timestamp': f"{yesterday}T07:00:00",
                    'sleep_score': int(np.clip(base_eff + np.random.normal(0,4), 40, 100)),
                    'total_sleep_h': 8.0,
                    'sleep_eff': base_eff,
                    'rem_pct': 18,
                    'deep_pct': 18,
                    'light_pct': 64 - (base_eff % 10),
                    'avg_hr': round(float(np.clip(62 + np.random.normal(0,3), 30, 110)),1),
                    'avg_rr': 14.0,
                    'avg_spo2': float(np.clip(97 + np.random.normal(0,0.8), 85, 100)),
                    'intervals': intervals,
                    'auto': True,
                    'locked': False,
                    'deleted': False,
                    'annotation': None,
                }
                st.session_state['sleep_history'].append(rec)
                append_audit('auto_recorded', user='system', target_id=rec['id'], meta={'date': yesterday})
                if persist_sleep:
                    save_sleep_history(st.session_state['sleep_history'])
                st.info(f"Auto-recorded sleep summary for {yesterday}")

    # (Hourly auto-snapshot removed.) Sleep detection will be used to create daily records automatically when the system detects a sleep session.

    # Admin mode removed — the app does not require an admin key. All users can edit their own history; doctors are read-only by design.
    if 'is_admin' not in st.session_state:
        st.session_state['is_admin'] = False

    # Doctor login (read-only) — accepts literal 'doctor' or DOCTOR_PASS env variable
    doc_key = st.sidebar.text_input('Doctor Key (optional, read-only)', type='password')
    if 'is_doctor' not in st.session_state:
        st.session_state['is_doctor'] = False
    if doc_key:
        if doc_key == 'doctor' or ('DOCTOR_PASS' in os.environ and doc_key == os.environ['DOCTOR_PASS']):
            st.session_state['is_doctor'] = True
            st.sidebar.success('Doctor access granted (read-only)')
        else:
            st.sidebar.error('Incorrect doctor key')

    # If doctor unlocked, show a button to view unedited sleep history
    if st.session_state.get('is_doctor', False):
        if st.sidebar.button('Show unedited sleep history', key='show_unedited_sleep_history'):
            st.session_state['doctor_view'] = True
            # Use st.stop() to end the current run and allow the updated flag to be displayed immediately
            st.stop()

    # Repository mode: when enabled by admin, nightly auto-records become locked and cannot be deleted by users
    # Repository mode (admin only). Changing this emits an audit entry.
    # Repository mode removed; records are editable by users and visible to doctors as read-only
    st.session_state['repo_mode'] = False

    # Load existing sleep history from disk if available and not already loaded
    if 'sleep_history' not in st.session_state:
        if persist_sleep:
            st.session_state['sleep_history'] = load_sleep_history()
        else:
            st.session_state['sleep_history'] = []

    # Doctor view: show a read-only unedited view when requested
    if st.session_state.get('doctor_view', False):
        st.header('Doctor — Unedited Sleep Records (Read-only)')
        # If in demo override and we haven't yet generated demo records for doctor, produce them
        def generate_demo_records(n=12):
            rng = np.random.default_rng(seed=42)
            demo_list = []
            for _ in range(n):
                base_eff = int(np.clip(70 + rng.normal(0,8), 30, 100))
                n_intervals = 16
                intervals = [{'index': i, 'quality': int(np.clip(base_eff + rng.normal(0,10), 20, 100))} for i in range(n_intervals)]
                rec = {
                    'id': str(uuid.uuid4()),
                    'timestamp': (datetime.now() - pd.Timedelta(days=int(rng.integers(1,60)))).isoformat(),
                    'sleep_score': int(np.clip(base_eff + rng.normal(0,6), 30, 100)),
                    'total_sleep_h': float(np.clip(6 + rng.normal(0,1.5), 3, 10)),
                    'sleep_eff': base_eff,
                    'rem_pct': int(np.clip(18 + rng.normal(0,6), 0, 60)),
                    'deep_pct': int(np.clip(18 + rng.normal(0,6), 0, 60)),
                    'light_pct': 100 - (int(np.clip(18 + rng.normal(0,6), 0, 60)) + int(np.clip(18 + rng.normal(0,6), 0, 60))),
                    'avg_hr': round(float(np.clip(60 + rng.normal(0,6), 30, 110)),1),
                    'avg_rr': round(float(np.clip(13 + rng.normal(0,3), 6, 30)),1),
                    'avg_spo2': round(float(np.clip(96 + rng.normal(0,1), 85, 100)),1),
                    'intervals': intervals,
                    'auto': True,
                    'locked': True,
                    'deleted': False,
                    'demo': True,
                }
                demo_list.append(rec)
            return demo_list

        if st.session_state.get('demo_override', False) and not st.session_state.get('doctor_demo_generated', False):
            st.session_state['sleep_history'].extend(generate_demo_records(n=15))
            st.session_state['doctor_demo_generated'] = True

        # Show the records in a simple table (no editing controls)
        df = pd.DataFrame(st.session_state.get('sleep_history', []))
        if df.shape[0] == 0:
            st.info('No records available.')
        else:
            st.dataframe(df.sort_values('timestamp', ascending=False).reset_index(drop=True))
        if st.button('Close doctor view', key='close_doctor_view'):
            st.session_state['doctor_view'] = False
            # Use st.stop() to end the current run and allow the updated flag to be displayed immediately
            st.stop()
        st.stop()

    # Deprecated: automatic morning auto-recording removed in favor of live sleep-detection; the system will record a daily entry when it detects a sleep session via sensors/vitals.

    # Simulate current sleep metrics (or show 'No user detected' when in real Anti-bedsore mode without demo)
    demo_mode = st.session_state.get('demo_override', False)
    # get current sensors to estimate presence
    heat_dp, presence_sensors = get_pressure_data(mode=device_mode, demo=demo_mode, massage_intensity=massage_intensity)
    presence_threshold = max(8.0, float(threshold_pressure) * 0.6)
    user_present = float(presence_sensors['mmHg'].max()) > presence_threshold

    # If the bed is empty (and not in demo mode), hide sleep metrics and zero vitals. In demo mode or when the
    # user is present, compute sleep metrics the same way for all device modes.
    if not demo_mode and not user_present:
        # Bed empty: sleep metrics not available; vitals zeroed
        sleep_score = None
        total_sleep_h = None
        sleep_eff = None
        rem_pct = None
        deep_pct = None
        light_pct = None
        avg_hr = 0.0
        avg_rr = 0.0
        avg_spo2 = 0.0
        # Zero session vitals so the top-row vitals display shows zeros
        st.session_state['vitals'] = {'hr': 0.0, 'spo2': 0.0, 'vo2': 0.0, 'rr': 0.0}
        allow_manual_record = False
        st.info('No user detected — sleep metrics unavailable while bed is empty; vitals set to 0.')
    else:
        # Demo or user present: generate demo/dummy sleep metrics
        rng = np.random.default_rng()
        sleep_score = int(np.clip(78 + rng.normal(0, 6), 40, 100))
        total_sleep_h = float(np.clip(7.0 + rng.normal(0, 0.8), 3.0, 12.0))
        sleep_eff = int(np.clip(88 + rng.normal(0, 6), 40, 100))
        rem_pct = int(np.clip(18 + rng.normal(0,4), 0, 60))
        deep_pct = int(np.clip(18 + rng.normal(0,4), 0, 60))
        light_pct = max(0, 100 - rem_pct - deep_pct)
        # Ensure session vitals are present and up-to-date so Sleep Tracking shows the same values
        if 'vitals' not in st.session_state:
            st.session_state['vitals'] = {'hr': 62.0, 'spo2': 98.0, 'vo2': 20.0, 'rr': 14.0}
        # If demo just started, apply immediate bump and then ensure 3s cadence updates are applied here
        demo_now = st.session_state.get('demo_override', False)
        now_ts_local = time.time()
        last_demo_local = st.session_state.get('last_demo_override', False)
        if demo_now and not last_demo_local:
            st.session_state['vitals']['hr'] = float(np.clip(st.session_state['vitals'].get('hr',62.0) + np.random.uniform(6.0,12.0), 48.0, 130.0))
            st.session_state['last_vitals_demo_update'] = now_ts_local
        if demo_now:
            last_u = st.session_state.get('last_vitals_demo_update', 0.0)
            if now_ts_local - last_u >= 3.0:
                st.session_state['vitals']['hr'] = float(np.clip(st.session_state['vitals'].get('hr',62.0) + np.random.normal(0.4,1.6), 48.0, 130.0))
                st.session_state['last_vitals_demo_update'] = now_ts_local
        st.session_state['last_demo_override'] = demo_now

        # Use session vitals for display so Dashboard and Sleep Tracking agree
        avg_hr = round(float(st.session_state['vitals'].get('hr', 62.0)), 1)
        avg_rr = round(float(st.session_state['vitals'].get('rr', 14.0)), 1)
        avg_spo2 = round(float(st.session_state['vitals'].get('spo2', 97.0)), 1)
        allow_manual_record = True

    # Show sleep summary metrics here (moved from Dashboard)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sleep Score", f"{sleep_score}" if sleep_score is not None else "N/A")
    c2.metric("Total Sleep", f"{total_sleep_h:.1f} h" if total_sleep_h is not None else "N/A")
    c3.metric("Sleep Efficiency", f"{sleep_eff}%" if sleep_eff is not None else "N/A")
    c4.metric("Avg HR", f"{avg_hr:.0f} bpm" if avg_hr is not None else "N/A")

    st.markdown("---")
    st.subheader("Stage Breakdown")
    s1, s2, s3 = st.columns(3)
    s1.metric("REM", f"{rem_pct}%" if rem_pct is not None else "N/A")
    s2.metric("Deep", f"{deep_pct}%" if deep_pct is not None else "N/A")
    s3.metric("Light", f"{light_pct}%" if light_pct is not None else "N/A")

    # Current risk snapshot (Alzheimer-like, Diabetes, Hypertension, Cardiovascular, Sleep Apnea)
    st.markdown('**Current Risk Snapshot:**')
    if sleep_eff is None or avg_hr is None or avg_spo2 is None or total_sleep_h is None:
        # No reliable data in real use mode — show N/A for all conditions
        na_rows = [{'condition': k, 'risk_pct': 'N/A'} for k in ['Alzheimer-like', 'Diabetes', 'Hypertension', 'Cardiovascular', 'Sleep Apnea']]
        st.table(pd.DataFrame(na_rows))
    else:
        cur_rec = {'sleep_eff': sleep_eff, 'avg_hr': avg_hr, 'avg_spo2': avg_spo2, 'total_sleep_h': total_sleep_h, 'rem_pct': rem_pct, 'deep_pct': deep_pct}
        risks_now = compute_risks(cur_rec)
        risk_df = pd.DataFrame([{'condition':k, 'risk_pct':f"{v:.0f}%"} for k,v in risks_now.items()])
        st.table(risk_df)




    st.markdown("---")
    st.write("Use the button to record a snapshot of the current simulated sleep metrics into history.")
    if st.button("Record Sleep Snapshot", key='record_sleep_snapshot'):
        if not allow_manual_record:
            st.warning('No user detected: cannot record a manual snapshot in real use mode.')
        else:
            # Build a richer record with intervals and metadata
            rec_id = str(uuid.uuid4())
            intervals = []
            n_intervals = max(1, math.ceil(total_sleep_h * 2))  # 30-min intervals
            for i in range(n_intervals):
                # Quality influenced by sleep_eff and some randomness
                q = int(np.clip(sleep_eff + np.random.normal(0,6), 20, 100))
                intervals.append({'index': i, 'quality': int(q)})

            entry = {
                'id': rec_id,
                'timestamp': datetime.now().isoformat(),
                'sleep_score': sleep_score,
                'total_sleep_h': total_sleep_h,
                'sleep_eff': sleep_eff,
                'rem_pct': rem_pct,
                'deep_pct': deep_pct,
                'light_pct': light_pct,
                'avg_hr': round(avg_hr,1),
                'avg_rr': round(avg_rr,1),
                'avg_spo2': round(avg_spo2,1),
                'intervals': intervals,
                'auto': False,
                'locked': False,
                'deleted': False,
                'annotation': None,
            }
            # append and persist
            st.session_state['sleep_history'].append(entry)
            append_audit('manual_record', user='user', target_id=rec_id, meta={'sleep_score': entry['sleep_score']})
            if persist_sleep:
                save_sleep_history(st.session_state['sleep_history'])
            st.success('Snapshot recorded and saved.')

            # Show immediate details for the newly created entry
            st.subheader('Recorded Snapshot Details')
            if len(entry.get('intervals', [])) > 0:
                int_df = pd.DataFrame(entry['intervals'])
                int_df['label'] = (int_df['index'] + 1).astype(str)
                chart = alt.Chart(int_df).mark_rect().encode(
                    x=alt.X('label:N', title='30-min Interval'),
                    y=alt.value(1),
                    color=alt.Color('quality:Q', scale=alt.Scale(domain=[20,100], scheme='redyellowgreen'), title='Quality')
                ).properties(height=60)
                st.altair_chart(chart, use_container_width=True)

            risks = compute_risks(entry)
            st.table(pd.DataFrame([{'condition':k, 'risk_pct':f"{v:.0f}%"} for k,v in risks.items()]))
            st.markdown('**Suggestions:**')
            for cond, pct in risks.items():
                if pct > 60:
                    st.write(f"- **High risk of {cond} ({pct:.0f}%)**: consider seeking medical advice.")
                elif pct > 30:
                    st.write(f"- Moderate risk of {cond} ({pct:.0f}%). Improve sleep hygiene and monitor.")
                else:
                    st.write(f"- Low risk of {cond} ({pct:.0f}%). Maintain good habits.")

            # Post-record actions (delete/lock/recover limited by roles)
            cols = st.columns([1,1,1,1])
        if cols[0].button('Soft-delete', key=f"sdel_{entry['id']}"):
            entry['deleted'] = True
            entry['deleted_at'] = datetime.now().isoformat()
            entry['deleted_by'] = 'user'
            save_sleep_history(st.session_state['sleep_history'])
            append_audit('deleted', user='user', target_id=entry['id'])
            st.success('Record soft-deleted')
        # Recovery / lock / permanent delete available to non-doctor users
        if not st.session_state.get('is_doctor', False):
            if cols[1].button('Recover', key=f'rec_{entry["id"]}'):
                entry['deleted'] = False
                entry.pop('deleted_at', None)
                entry['deleted_by'] = None
                save_sleep_history(st.session_state['sleep_history'])
                append_audit('recovered', user='user', target_id=entry['id'])
                st.success('Record recovered')
            lock_btn = 'Unlock' if entry.get('locked', False) else 'Lock'
            if cols[2].button(lock_btn, key=f"lockrec_{entry['id']}"):
                entry['locked'] = not entry.get('locked', False)
                save_sleep_history(st.session_state['sleep_history'])
                append_audit('locked' if entry['locked'] else 'unlocked', user='user', target_id=entry['id'])
                st.success(f"Record {'locked' if entry.get('locked') else 'unlocked'}")
            if cols[3].button('Permanently Delete', key=f'permdel_{entry["id"]}'):
                st.session_state['sleep_history'] = [r for r in st.session_state['sleep_history'] if r['id'] != entry['id']]
                save_sleep_history(st.session_state['sleep_history'])
                append_audit('permanently_deleted', user='user', target_id=entry['id'])
                st.success('Record permanently deleted')
        else:
            cols[1].write('Doctor: read-only (no recover/delete)')

        # Toggle to show deleted records
        show_deleted = st.sidebar.checkbox('Show deleted records', value=False, key='show_deleted_records')

        # Download CSV of visible (non-deleted unless 'show deleted' selected) records
        visible = [r for r in st.session_state['sleep_history'] if not r.get('deleted', False) or show_deleted]
        if len(visible) > 0:
            csv = pd.DataFrame(visible).to_csv(index=False)
            st.download_button("Download sleep history CSV", data=csv, file_name="sleep_history.csv", mime="text/csv")
        if persist_sleep and os.path.exists(SLEEP_HISTORY_JSON_PATH):
            st.caption(f"Saved to {SLEEP_HISTORY_JSON_PATH}")

        # --- Record Viewer & Annotation Editor ---
        visible_records = [r for r in st.session_state['sleep_history'] if not r.get('deleted', False) or show_deleted]
        visible_records_sorted = sorted(visible_records, key=lambda r: r.get('timestamp',''), reverse=True)
        options = [f"{r.get('timestamp','')[:10]} - {r.get('id')[:8]} ({'AUTO' if r.get('auto') else 'MANUAL'})" for r in visible_records_sorted]
        selected = None
        if len(options) > 0:
            sel = st.selectbox('Select a record to view / annotate', options, key='select_sleep_record')
            idx = options.index(sel)
            selected = visible_records_sorted[idx]

        if selected is not None:
            rec = selected
            st.subheader(f"Viewing record: {rec.get('timestamp','')}  {rec.get('id')}")
            # Show intervals visualization
            intervals = rec.get('intervals', [])
            if intervals:
                int_df = pd.DataFrame(intervals)
                int_df['label'] = (int_df['index'] + 1).astype(str)
                chart = alt.Chart(int_df).mark_rect().encode(
                    x=alt.X('label:N', title='30-min Interval'),
                    y=alt.value(1),
                    color=alt.Color('quality:Q', scale=alt.Scale(domain=[20,100], scheme='redyellowgreen'), title='Quality')
                ).properties(height=60)
                st.altair_chart(chart, use_container_width=True)

            # Show disease risks
            risks = compute_risks(rec)
            st.table(pd.DataFrame([{'condition':k, 'risk_pct':f"{v:.0f}%"} for k,v in risks.items()]))

            # Annotation editor
            st.markdown('**Annotation (editable)**')
            ann_key = f"annotation_{rec['id']}"
            existing_ann = rec.get('annotation', '') if rec.get('annotation') is not None else ''
            ann_text = st.text_area('Edit annotation', value=existing_ann, key=ann_key)
            if st.button('Save Annotation', key=f'save_ann_{rec["id"]}'):
                rec['annotation'] = ann_text
                save_sleep_history(st.session_state['sleep_history'])
                append_audit('save_annotation', user=('doctor' if st.session_state.get('is_doctor') else 'user'), target_id=rec['id'], meta={'annotation': ann_text})
                st.success('Annotation saved and audited')

            # Record actions with permission checks
            cols = st.columns([1,1,1,1])
            # Delete: allowed for users but not doctors; locked records cannot be deleted
            can_delete = (not rec.get('locked', False)) and (not st.session_state.get('is_doctor', False))
            if cols[0].button('Soft-delete', key=f"sdel_{rec['id']}", disabled=not can_delete):
                rec['deleted'] = True
                rec['deleted_at'] = datetime.now().isoformat()
                rec['deleted_by'] = 'user'
                save_sleep_history(st.session_state['sleep_history'])
                append_audit('deleted', user='user', target_id=rec['id'])
                st.success('Record soft-deleted')

            # Recover and permanent delete available to non-doctor users
            if not st.session_state.get('is_doctor', False):
                if cols[1].button('Recover', key=f'rec_{rec["id"]}'):
                    rec['deleted'] = False
                    rec.pop('deleted_at', None)
                    rec['deleted_by'] = None
                    save_sleep_history(st.session_state['sleep_history'])
                    append_audit('recovered', user='user', target_id=rec['id'])
                    st.success('Record recovered')
                # Lock/unlock to prevent edits
                lock_btn = 'Unlock' if rec.get('locked', False) else 'Lock'
                if cols[2].button(lock_btn, key=f"lockrec_{rec['id']}"):
                    rec['locked'] = not rec.get('locked', False)
                    save_sleep_history(st.session_state['sleep_history'])
                    append_audit('locked' if rec['locked'] else 'unlocked', user='user', target_id=rec['id'])
                    st.success(f"Record {'locked' if rec.get('locked') else 'unlocked'}")
                if cols[3].button('Permanently Delete', key=f'permdel_{rec["id"]}'):
                    st.session_state['sleep_history'] = [r for r in st.session_state['sleep_history'] if r['id'] != rec['id']]
                    save_sleep_history(st.session_state['sleep_history'])
                    append_audit('permanently_deleted', user='user', target_id=rec['id'])
                    st.success('Record permanently deleted')
            else:
                cols[1].write('Doctor: read-only (no recover/delete)')

        # --- Audit Log Viewer (doctor only) ---
        if st.session_state.get('is_doctor', False):
            st.markdown('---')
            st.subheader('Audit Log')
            logs = load_audit_logs()
            if len(logs) == 0:
                st.info('No audit logs found yet.')
            else:
                # show recent logs and allow download
                last_n = st.number_input('Show last N entries', min_value=5, max_value=5000, value=25, step=5)
                display_logs = sorted(logs, key=lambda e: e.get('timestamp',''), reverse=True)[:int(last_n)]
                st.table(pd.DataFrame(display_logs))
                st.download_button('Download audit log (JSON)', data=json.dumps(logs, indent=2), file_name='audit_log.json', mime='application/json')

        # --- Risk Trends (aggregate) ---
        if len(st.session_state['sleep_history']) > 0:
            st.markdown('---')
            st.subheader('Risk Trends')
            # Build DataFrame with date and risks
            rows = []
            for r in st.session_state['sleep_history']:
                if r.get('deleted', False):
                    continue
                try:
                    date = r.get('timestamp','')[:10]
                except Exception:
                    date = str(r.get('timestamp',''))[:10]
                risks = compute_risks(r)
                row = {'date': date}
                row.update(risks)
                rows.append(row)
            if len(rows) > 0:
                df_r = pd.DataFrame(rows)
                df_long = df_r.melt(id_vars=['date'], var_name='condition', value_name='risk')
                cond_choice = st.selectbox('Select condition', options=sorted(df_long['condition'].unique()), index=0)
                sub = df_long[df_long['condition'] == cond_choice]
                sub['date'] = pd.to_datetime(sub['date'])
                line = alt.Chart(sub).mark_line(point=True).encode(x='date:T', y='risk:Q').properties(height=200)
                st.altair_chart(line, use_container_width=True)
                # allow CSV download of aggregated risk trends
                csv_report = df_r.to_csv(index=False)
                st.download_button('Download risk trends CSV', data=csv_report, file_name='risk_trends.csv', mime='text/csv')


    # --- Chat / Sleep Coach panel ---
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader('Sleep Coach Chat')
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        # Render chat messages in a fixed-height scrollable box
        chat_html = """
        <div style='height:320px; overflow:auto; background-color:#0d1117; padding:12px; border-radius:8px; border:1px solid #222;'>
        """
        for msg in st.session_state['chat_history']:
            who = msg['role']
            text = msg['text'].replace('\n', '<br>')
            color = '#9ad1ff' if who.lower() == 'coach' else '#dcdcdc'
            chat_html += f"<div style='margin-bottom:8px;'><strong style='color:{color};'>{who}</strong>: <span style='color:#eee'>{text}</span></div>"
        chat_html += "</div>"
        # Wrap the chat HTML inside a uniquely identifiable container so we can reliably remove it
        chat_container = st.container()
        chat_container.markdown(f"<div id='sleep_coach_container'>{chat_html}</div>", unsafe_allow_html=True)

        # Free-text coach input disabled — use the FAQ dropdown below for common questions.
        st.caption('Free-text coach input is currently disabled. Use the FAQ dropdown or the Coach Tips panel for guidance.')

        # --- FAQ helper (20 common sleep questions) ---
        faq_list = [
            'How much sleep do adults need?',
            'What is good sleep hygiene?',
            'How can I fall asleep faster?',
            'When should I see a doctor for insomnia?',
            'Does alcohol help or harm sleep?',
            'How does caffeine affect sleep and when to stop?',
            'Is napping okay and how long should I nap?',
            'How does exercise affect sleep?',
            'What bedroom temperature is best for sleep?',
            'How do screens and blue light affect my sleep?',
            'How can I improve deep sleep?',
            'What causes frequent awakenings at night?',
            'How are sleep and weight related?',
            'What is sleep apnea and common signs?',
            'Are sleep medications safe?',
            'How to manage shift work sleep issues?',
            'How does stress or anxiety affect sleep?',
            'How accurate are consumer sleep trackers?',
            'How long should I wait before going to bed if I can’t sleep?',
            'What is the difference between REM and deep sleep?'
        ]
        faq_choice = st.selectbox('Pick a common question (FAQ)', options=faq_list, index=0)
        if st.button('Ask FAQ', key='ask_faq'):
            # canned short answers for the FAQ list
            faq_answers = {
                'How much sleep do adults need?': 'Most adults need 7–9 hours per night. Individual needs vary; aim for consistent duration and timing.',
                'What is good sleep hygiene?': 'Stick to a regular sleep schedule, avoid caffeine/alcohol near bedtime, keep the bedroom dark/quiet/cool, and wind down before bed.',
                'How can I fall asleep faster?': 'Use a short relaxation routine, avoid screens for 30–60 minutes before bed, and avoid trying too hard—get up and do a quiet activity if unable to sleep.',
                'When should I see a doctor for insomnia?': 'If poor sleep persists for weeks despite good sleep hygiene or if it impairs daytime function, seek a clinician or sleep specialist.',
                'Does alcohol help or harm sleep?': 'Alcohol may make you fall asleep faster but fragments sleep later and reduces REM/deep sleep—overall harms sleep quality.',
                'How does caffeine affect sleep and when to stop?': 'Caffeine can disturb sleep for many hours; avoid it after mid-afternoon (or 6–8 hours before bedtime).',
                'Is napping okay and how long should I nap?': 'Short naps (10–30 min) can be restorative; long naps can interfere with nighttime sleep for some people.',
                'How does exercise affect sleep?': 'Regular daytime exercise improves sleep, but vigorous exercise right before bed may be stimulating for some people.',
                'What bedroom temperature is best for sleep?': 'A cool room (about 16–19°C / 60–67°F) supports better sleep for most adults.',
                'How do screens and blue light affect my sleep?': 'Blue light suppresses melatonin and can delay sleep—use night modes and reduce screen time before bed.',
                'How can I improve deep sleep?': 'Consistent sleep schedule, regular exercise, and reducing alcohol/caffeine can increase deep sleep over time.',
                'What causes frequent awakenings at night?': 'Stress, sleep apnea, restless legs, noise, alcohol, and nighttime bathroom trips are common causes.',
                'How are sleep and weight related?': 'Poor sleep disrupts appetite hormones and metabolism and can increase risk of weight gain over time.',
                'What is sleep apnea and common signs?': 'Sleep apnea is breathing interruption during sleep; signs include loud snoring, gasping, and daytime sleepiness.',
                'Are sleep medications safe?': 'Short-term use can help but many have side effects; discuss risks/benefits with a clinician—behavioral therapies are preferred long-term.',
                'How to manage shift work sleep issues?': 'Use strategic naps, bright light exposure during waking shift, and darkness/sleep hygiene for daytime sleep; consider chronotherapy advice from specialists.',
                'How does stress or anxiety affect sleep?': 'Stress increases arousal and racing thoughts, making it hard to fall and stay asleep—relaxation training helps.',
                'How accurate are consumer sleep trackers?': 'They estimate sleep stages and duration but can be inaccurate—use trends rather than single-night values.',
                'How long should I wait before going to bed if I can’t sleep?': 'If unable to sleep after 20 minutes, get up and do a quiet activity until you feel sleepy, then return to bed.',
                'What is the difference between REM and deep sleep?': 'Deep (slow-wave) sleep is restorative and physical repair-focused; REM is important for memory and emotional processing. Both are important.'
            }
            reply = faq_answers.get(faq_choice, "I'm sorry, I don't have a short answer for that right now.")
            st.session_state['chat_history'].append({'role':'User', 'text': faq_choice})
            st.session_state['chat_history'].append({'role':'Coach', 'text': reply})
            try:
                append_audit('faq_asked', user=('doctor' if st.session_state.get('is_doctor') else 'user'), target_id=None, meta={'question': faq_choice})
            except Exception:
                pass


        # Free-text Send functionality removed per user request. Use the FAQ dropdown or Coach Tips to populate chat history.

    with c2:
        st.subheader('Coach Tips')
        st.write('- Keep consistent bedtime and wake time')
        st.write('- Reduce evening caffeine and screen time')
        st.write('- Keep room dark and cool')
        st.write('- If symptoms persist, consult a clinician')

    st.stop()

# Fetch Data (grid for visualization + sensors list for raw values)
heatmap_df, sensors_df = get_pressure_data(mode=device_mode, demo=st.session_state.get('demo_override', False), massage_intensity=massage_intensity)
# Note: Sensors remain active and report values even when bladders are emergency-deflated.
# Do NOT zero sensor readings or heatmap here; keep pressure trend intact so diagnostics remain useful.

# --- Compact Health Metrics (Dashboard) ---
# Sleep metrics have been relocated to the Sleep Tracking page; no dashboard display here.

current_max_pressure = sensors_df['mmHg'].max()
current_avg_pressure = sensors_df['mmHg'].mean()

# --- Maintain pressure history (1-min window) ---
if 'pressure_history' not in st.session_state:
    st.session_state['pressure_history'] = []
st.session_state['pressure_history'].append(float(current_max_pressure))
# keep last 120 samples to allow some smoothing (we use 60 for 1 minute)
st.session_state['pressure_history'] = st.session_state['pressure_history'][-120:]
avg_1min = float(np.mean(st.session_state['pressure_history'][-60:])) if len(st.session_state['pressure_history'])>=1 else float(current_max_pressure)

# --- SENSOR DIAGNOSTICS (stable placement) ---
# Place a placeholder here so widget ordering is stable across reruns and emergency actions
_diag_placeholder = st.empty()
with _diag_placeholder:
    with st.expander("🛠️ Technician / Sensor Diagnostics (Click to Expand)", expanded=False):
        st.write("Raw Sensor Data converted to Clinical Units (mmHg)")

        # Diagnostic render instrumentation (helps detect duplicates): increment counter and show render timestamp
        st.session_state['diagnostics_render_count'] = st.session_state.get('diagnostics_render_count', 0) + 1
        try:
            ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        except Exception:
            ts = str(time.time())
        st.caption(f"Render #{st.session_state['diagnostics_render_count']} at {ts}")

        # Display raw sensor readings for the 7 prototype sensors (live)
        st.table(sensors_df)

        # Also show per-sensor accumulated exposure (if available)
        if 'sensor_exposure' in st.session_state:
            exposures_df = pd.DataFrame(list(st.session_state['sensor_exposure'].items()), columns=['sensor_id', 'mmHg_min'])
            st.subheader("Per-sensor cumulative exposure (mmHg·min)")
            st.table(exposures_df)

        st.caption("Calibration Formula: mmHg = (ADC_Value * 0.045) - 12.5")

# Determine Risk using 1-minute average and hysteresis to avoid rapid flips
if 'risk_state' not in st.session_state:
    st.session_state['risk_state'] = 'SAFE'
    st.session_state['risk_counter'] = 0

# Use avg_1min for smoother decision making
warning_thresh = max(30.0, float(threshold_pressure))
critical_thresh = 40.0
if avg_1min > critical_thresh:
    candidate_state = 'CRITICAL'
elif avg_1min > warning_thresh:
    candidate_state = 'WARNING'
else:
    candidate_state = 'SAFE'

# Hysteresis: require the candidate state to persist for 3 consecutive checks
if candidate_state != st.session_state['risk_state']:
    st.session_state['risk_counter'] += 1
else:
    st.session_state['risk_counter'] = 0

if st.session_state['risk_counter'] >= 3:
    st.session_state['risk_state'] = candidate_state
    st.session_state['risk_counter'] = 0

risk_state = st.session_state['risk_state']
if risk_state == 'CRITICAL':
    risk_color = 'inverse'
elif risk_state == 'WARNING':
    risk_color = 'off'
else:
    risk_color = 'normal'

# --- ROW 1: KEY METRICS (Modified Visuals) ---
col1, col2, col3, col4 = st.columns(4)
# Show 1-minute average in the delta for context (helps user understand recent trend)
col1.metric("Patient Risk Status", risk_state, delta=f"1-min avg: {avg_1min:.1f} mmHg", delta_color=risk_color)
col2.metric("Peak Pressure", f"{current_max_pressure:.1f} mmHg", f"{current_max_pressure-threshold_pressure:.1f} vs Limit")
col3.metric("Avg Surface Pressure", f"{current_avg_pressure:.1f} mmHg")

# --- VITALS (replace Next Auto-Turn) ---
# Smooth vitals across reruns
if 'vitals' not in st.session_state:
    st.session_state['vitals'] = {'hr': 62.0, 'spo2': 98.0, 'vo2': 20.0, 'rr': 14.0}

v = st.session_state['vitals']
# Determine user presence now so vitals can be zeroed in real modes when no user is present
demo_mode = st.session_state.get('demo_override', False)
try:
    _, presence_sensors_v = get_pressure_data(mode=device_mode, demo=demo_mode, massage_intensity=massage_intensity)
    presence_threshold_v = max(8.0, float(threshold_pressure) * 0.6)
    user_present_v = float(presence_sensors_v['mmHg'].max()) > presence_threshold_v
except Exception:
    user_present_v = True  # default to present if detection fails

# If running in real anti-bedsore or massage modes and no user is present, set vitals to zero
if device_mode in ('Anti-bedsore', 'Massage Mode') and not demo_mode and not user_present_v:
    v['hr'] = 0.0
    v['spo2'] = 0.0
    v['vo2'] = 0.0
    v['rr'] = 0.0
else:
    # Vitals update behavior differs when running in demo mode vs real mode.
    # Demo mode: apply an immediate bump when demo starts and then update vitals every 3 seconds
    demo_mode_local = st.session_state.get('demo_override', False)
    now_ts = time.time()
    # detect demo start transition
    last_demo = st.session_state.get('last_demo_override', False)
    if demo_mode_local and not last_demo:
        # immediate visible bump so user sees the demo take effect
        v['hr'] = float(np.clip(v['hr'] + float(np.random.uniform(6.0, 12.0)), 48.0, 130.0))
        st.session_state['last_vitals_demo_update'] = now_ts
    if demo_mode_local:
        last_update = st.session_state.get('last_vitals_demo_update', 0.0)
        # only update every 3 seconds to produce a visible cadence
        if now_ts - last_update >= 3.0:
            # stronger, more noticeable random walk during demo
            v['hr'] = float(np.clip(v['hr'] + np.random.normal(0.4, 1.6), 48.0, 130.0))
            v['spo2'] = float(np.clip(v['spo2'] + np.random.normal(0, 0.2), 85.0, 100.0))
            v['vo2'] = float(np.clip(v['vo2'] + np.random.normal(0, 0.4), 10.0, 40.0))
            v['rr'] = float(np.clip(v['rr'] + np.random.normal(0, 0.5), 8.0, 30.0))
            st.session_state['last_vitals_demo_update'] = now_ts
    else:
        # Normal, subtle random walk with caps to avoid sudden jumps
        v['hr'] = float(np.clip(v['hr'] + np.random.normal(0, 0.6), 40, 120))
        v['spo2'] = float(np.clip(v['spo2'] + np.random.normal(0, 0.1), 85, 100))
        v['vo2'] = float(np.clip(v['vo2'] + np.random.normal(0, 0.2), 10.0, 40.0))
        v['rr'] = float(np.clip(v['rr'] + np.random.normal(0, 0.3), 8.0, 30.0))
    # persist last_demo_override
    st.session_state['last_demo_override'] = demo_mode_local

# Update sleep detector with fresh readings (this will create a daily record when a sleep session ends)
try:
    update_sleep_state(sensors_df, avg_1min, v)
except Exception:
    pass

# Heart Rate metric: update every rerun so changes are visible immediately
col4.metric("Heart Rate", f"{v['hr']:.0f} bpm", delta=f"SpO2: {v['spo2']:.0f}%")
# Additional small metrics row
vm1, vm2, vm3 = st.columns(3)
vm1.metric("VO2", f"{v['vo2']:.1f} mL/kg/min")
vm2.metric("Resp Rate", f"{v['rr']:.0f} brpm")
vm3.metric("SpO2", f"{v['spo2']:.0f}%")

st.markdown("---")

# --- ROW 2: VISUALIZATION (Heatmap + Chart) ---
# Automatic bladder control: map sensor pressures to 4 bladder targets and update targets smoothly.
# Mapping: LT = mean(S1,S4), RT = mean(S2,S5), LB = mean(S6,S3), RB = mean(S7,S3)
# If a trained bladder policy exists in the loaded model, it can override the heuristic.
left_col, right_col = st.columns([1, 2])

# compute automatic desired bladder targets based on sensors
try:
    vals = list(sensors_df['mmHg'].values)
    s1,s2,s3,s4,s5,s6,s7 = [float(v) for v in vals]
    lt_p = np.mean([s1, s4])
    rt_p = np.mean([s2, s5])
    lb_p = np.mean([s6, s3])
    rb_p = np.mean([s7, s3])
    local_pressures = np.array([lt_p, rt_p, lb_p, rb_p], dtype=float)
except Exception:
    local_pressures = np.array([0.0,0.0,0.0,0.0], dtype=float)

# baseline target height
base_h = 5.0
# scale factor: how many cm per mmHg difference from threshold
scale_k = 0.08
# heuristic desired targets
desired = base_h + (float(threshold_pressure) - local_pressures) * scale_k
# Check for an ML-based bladder policy in the loaded model (optional)
model = st.session_state.get('ml_model')
if isinstance(model, dict):
    try:
        # Prefer a learned linear policy (coef + intercept)
        if 'bladder_policy_model' in model:
            coef = np.array(model['bladder_policy_model']['coef'])  # (7,4)
            intercept = np.array(model['bladder_policy_model']['intercept'])  # (4,)
            try:
                Xcur = np.array(list(sensors_df['mmHg'].astype(float).values))  # shape (7,)
                pred = Xcur @ coef + intercept  # (4,)
                # blend learned policy with heuristic
                desired = 0.5 * desired + 0.5 * pred
            except Exception:
                pass
        elif model.get('bladder_policy'):
            bp = np.array(model.get('bladder_policy'))
            if bp.shape[0] == 4:
                desired = 0.6 * desired + 0.4 * bp
    except Exception:
        pass

# clip to valid bladder height range (system enforces 2–8 cm actuator range)
desired = np.clip(desired, 2.0, 8.0)

# Enforce targets from the automatic policy (unless emergency deflated)
if st.session_state.get('emergency_deflated', False):
    st.session_state['bladder_targets'] = [0.0, 0.0, 0.0, 0.0]
else:
    st.session_state['bladder_targets'] = list(desired.round(2))

# rate-limit how fast targets can change (cm per rerun approximation)
rate_mult = 1.0
if device_mode == 'Massage Mode':
    rate_mult = 1.0 + float(massage_intensity) * 2.0  # more intensity -> faster change
max_target_delta = float(max_bladder_speed) * rate_mult
current_targets = np.array(st.session_state.get('bladder_targets', [base_h]*4), dtype=float)
delta_t = desired - current_targets
applied = np.clip(delta_t, -max_target_delta, max_target_delta)
new_targets = np.clip(current_targets + applied, 0.0, 8.0)
# If emergency deflated, keep targets at zero (will be restored elsewhere)
if st.session_state.get('emergency_deflated', False):
    st.session_state['bladder_targets'] = [0.0, 0.0, 0.0, 0.0]
else:
    st.session_state['bladder_targets'] = list(new_targets.round(2))

with left_col:
    # Mattress header and caption removed for a cleaner layout
    st.empty()
if 'emergency_deflated' not in st.session_state:
    st.session_state['emergency_deflated'] = False

# Smoothly adjust current bladder heights toward targets using max_bladder_speed (cm/sec)
# If emergency deflation is active, animate a controlled deflation to 0 cm over 10 seconds
if 'bladder_current' not in st.session_state:
    st.session_state['bladder_current'] = list(st.session_state['bladder_targets'])

if st.session_state.get('emergency_deflated', False):
    # perform a timed deflation animation based on recorded 'bladder_current_pre_emergency' and start timestamp
    pre = np.array(st.session_state.get('bladder_current_pre_emergency', st.session_state.get('bladder_current', [5.0,5.0,5.0,5.0])), dtype=float)
    start = st.session_state.get('emergency_deflate_started_at', None)
    if start is None:
        # if missing, initialize now
        st.session_state['emergency_deflate_started_at'] = time.time()
        start = st.session_state['emergency_deflate_started_at']
    elapsed = max(0.0, time.time() - start)
    frac = min(1.0, elapsed / 10.0)
    new_current = np.clip(pre * (1.0 - frac), 0.0, 8.0)
    st.session_state['bladder_current'] = list(new_current.round(1))
    current_heights = np.array(st.session_state['bladder_current'], dtype=float)
else:
    # No manual input application — targets controlled by policy (handled above).
    # Smoothly adjust current bladder heights toward targets using max_bladder_speed (cm/sec)

    targets = np.array(st.session_state['bladder_targets'], dtype=float)
    current = np.array(st.session_state['bladder_current'], dtype=float)
    # max change per rerun approximates per-second movement
    # Massage intensity affects effective speed
    if device_mode == 'Massage Mode':
        eff_speed = float(max_bladder_speed) * (1.0 + float(massage_intensity) * 2.0)
    else:
        eff_speed = float(max_bladder_speed)
    delta = targets - current
    change = np.clip(delta, -eff_speed, eff_speed)
    new_current = np.clip(current + change, 0.0, 8.0)
    st.session_state['bladder_current'] = list(new_current.round(1))
    current_heights = np.array(st.session_state['bladder_current'], dtype=float)

    # Render HTML grid showing bladder fill levels
    def render_bladders_html(heights):
        colors = ['#60a5fa','#fb7185','#34d399','#fbbf24']
        labels = ['Left Top','Right Top','Left Bottom','Right Bottom']
        html = '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:18px;max-width:540px;margin:auto;">'
        for i,h in enumerate(heights):
            pct = int((h / 8.0) * 100)
            color = colors[i]
            html += f"<div style='border:1px solid #333;border-radius:8px;padding:8px;background:#0f1720;color:#e6eef8;'>"
            html += "<div style='height:220px;background:#071022;display:flex;align-items:flex-end;justify-content:center;padding:8px;'>"
            html += f"<div style='width:72%;height:{pct}%;background:linear-gradient(180deg,{color},#0b1220);border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.5);display:flex;align-items:flex-end;justify-content:center;'></div>"
            html += "</div>"
            html += f"<div style='text-align:center;color:#ddd;margin-top:8px'>{labels[i]}<br/><strong>{h} cm</strong></div>"
            html += "</div>"
        html += '</div>'
        return html

    # Compact bladder numeric metrics removed to streamline the mattress panel layout


st.subheader("📉 Pressure Trend (live)")

# --- History (persist across reruns) ---
# (pressure_history is maintained earlier so nothing to append here)
# Build a timestamped DataFrame for plotting
history = st.session_state['pressure_history']
chart_df = pd.DataFrame({
    'timestamp': pd.date_range(end=datetime.now(), periods=len(history), freq='S'),
    'mmHg': history
})

# Chart color: red for high pressure
chart_color = "#ff4b4b" if current_max_pressure > 35 else "#00acee"
# If no history, avoid errors
if len(history) == 0:
    st.info("No pressure history yet; run the dashboard to collect samples.")
    chart_df = pd.DataFrame({'timestamp':[], 'mmHg':[]})

# Prefer Altair for clearer visualization (threshold line + tooltip)
try:
    base = alt.Chart(chart_df).encode(x=alt.X('timestamp:T', title='Time'))
    area = base.mark_area(opacity=0.2, color=chart_color).encode(y=alt.Y('mmHg:Q', title='Pressure (mmHg)'))
    line = base.mark_line(color=chart_color).encode(y='mmHg:Q')
    last_point = chart_df.tail(1)
    last = alt.Chart(last_point).mark_circle(size=80, color='white', stroke=chart_color, strokeWidth=2).encode(
        x='timestamp:T', y='mmHg:Q', tooltip=[alt.Tooltip('timestamp:T', title='Time'), alt.Tooltip('mmHg:Q', title='mmHg')]
    )
    threshold = 30
    thresh_rule = alt.Chart(pd.DataFrame({'y':[threshold]})).mark_rule(color='orange', strokeDash=[4,4]).encode(y='y:Q')
    chart = (area + line + last + thresh_rule).properties(height=220, width='container')
    st.altair_chart(chart, use_container_width=True)
except Exception:
    st.line_chart(chart_df.set_index('timestamp')['mmHg'], height=220)

    # --- Prediction / AI placeholder with exposure accumulation ---
# Controls: simulate no movement or reset exposure
try:
    simulate_hold = st.sidebar.checkbox("Simulate No Movement (accelerate stage 1)", value=False)
    if st.sidebar.button("Reset Exposure / Mark Moved"):
        if 'sensor_exposure' in st.session_state:
            del st.session_state['sensor_exposure']
        st.success("Exposure and counters reset")

    # Delta minutes per rerun: fast-forward if simulating, otherwise assume 1 second (~0.0167 min)
    delta_minutes = 5.0 if simulate_hold else (1.0/60.0)

    # Accumulate exposure using the simple model (uses user-tunable `threshold_pressure`)
    total_exposure, exposures = accumulate_exposure(sensors_df, delta_minutes=delta_minutes, threshold=threshold_pressure)

    # Compute current per-minute excess (mmHg per minute) across sensors using `threshold_pressure`
    per_minute_excess = sum(max(0.0, float(r['mmHg']) - threshold_pressure) for _, r in sensors_df.iterrows())

    # Predict hours using chosen model
    if 'model_choice' in locals() and model_choice in ("Demo Linear Model", "Upload JSON Model"):
        model = st.session_state.get('ml_model', None)
        if model is not None:
            est_hours = predict_with_model(model, sensors_df, total_exposure, per_minute_excess)
        else:
            est_hours = float('inf')
            st.sidebar.info("No ML model loaded — using heuristic until model is provided.")
    else:
        est_hours = predict_time_to_stage1_from_exposure(total_exposure, per_minute_excess, exposure_threshold=exposure_threshold)

    # Format display text
    if est_hours == float('inf'):
        est_text = ">72 h (no immediate risk)"
        est_display = None
    else:
        adjusted_hours = max(0.0, est_hours)
        if adjusted_hours >= 24:
            est_text = f"{adjusted_hours/24:.1f} days"
        else:
            est_text = f"{adjusted_hours:.1f} h"
        est_display = adjusted_hours

    # If running in real Anti-bedsore mode with no user present, show N/A for time/exposure (include Massage Mode)
    presence_threshold_dashboard = max(8.0, float(threshold_pressure) * 0.6)
    no_user_dashboard = (device_mode in ('Anti-bedsore','Massage Mode')) and (not st.session_state.get('demo_override', False)) and (float(sensors_df['mmHg'].max()) <= presence_threshold_dashboard)
    if no_user_dashboard:
        est_display = None
        est_text = 'N/A (no user)'
        total_exposure = 0.0
        per_minute_excess = 0.0
except Exception as e:
    # Fail safely: ensure UI continues rendering and show N/A where appropriate
    st.sidebar.error(f'Exposure estimation failed: {str(e)}')
    est_hours = float('inf')
    est_text = 'N/A (error)'
    est_display = None
    total_exposure = 0.0
    per_minute_excess = 0.0

# Split layout: left shows bladders; right shows estimated time + exposure
col_left, col_right = st.columns([2,1])

# Left: show canonical bladder diagram compactly aligned with title
with col_left:
    st.markdown("**Bladders**")
    # Stable placeholder ensures we update a single widget rather than creating new ones
    bladder_pl = st.empty()
    try:
        # Instrumentation: count renders and show timestamp to detect duplicates/stale renders
        st.session_state['bladder_render_count'] = st.session_state.get('bladder_render_count', 0) + 1
        try:
            b_ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        except Exception:
            b_ts = str(time.time())
        bladder_pl.caption(f"Bladder render #{st.session_state['bladder_render_count']} at {b_ts}")
        bladder_pl.markdown(render_bladders_html(current_heights), unsafe_allow_html=True)
    except Exception:
        bladder_pl.info("Bladder diagram unavailable")

# Client-side cleanup: remove duplicate "Bladders" sections (keep the first). Helps cases where a stale snapshot
# remains visible after emergency actions. This is a UI-only fix that removes the duplicate DOM nodes in-browser.
components.html('''
<script>
(function(){
  function removeDup(){
    const matches = [];
    document.querySelectorAll('*').forEach(el=>{
      if(!el) return;
      // look for leaf nodes whose trimmed text is exactly 'Bladders'
      if(el.childElementCount===0 && el.innerText && el.innerText.trim()==='Bladders') matches.push(el);
    });
    if(matches.length<=1) return;
    for(let i=1;i<matches.length;i++){
      let node = matches[i];
      // ascend a few levels to hide the whole section container
      for(let j=0;j<6 && node;j++) node = node.parentElement;
      if(node) node.style.display='none';
    }
  }
  // run a few times to catch dynamic updates
  setTimeout(removeDup, 100);
  let runs=0; let iv=setInterval(()=>{ removeDup(); if(++runs>12) clearInterval(iv); }, 200);
})();
</script>
''', height=0)

# Right: Est time and exposure
with col_right:
    try:
        st.markdown("**Estimated Time to Stage I if unmoved (demo, exposure-based)**")
        if est_display is None:
            st.metric("Est. Time to Stage I", est_text, delta=None)
        else:
            minutes_to_stage = est_hours * 60
            st.metric("Est. Time to Stage I", est_text, delta=f"{int(minutes_to_stage)} min to Stage I")

        st.markdown("**Cumulative Exposure (mmHg·min)**")
        st.metric("Total Exposure", f"{total_exposure:.0f} mmHg·min", delta=None)
        progress_val = min(1.0, total_exposure / exposure_threshold)
        st.progress(progress_val)

        st.caption("This is a demo model. Replace `accumulate_exposure` and `predict_time_to_stage1_from_exposure` with your ML-based logic later.")
    except Exception:
        st.info('Estimated time/exposure unavailable')



# --- AUTO REFRESH ---
time.sleep(1)
st.rerun()