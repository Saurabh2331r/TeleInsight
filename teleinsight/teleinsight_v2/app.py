from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd, os, traceback
from werkzeug.utils import secure_filename
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.ml_models import run_churn_prediction, run_segmentation, get_dashboard_stats

app = Flask(__name__)
app.secret_key = 'teleinsight_v2_dti_2026'
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED = {'csv','xlsx','xls'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

def allowed(f): return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED

def load_df():
    p = session.get('data_path')
    if p and os.path.exists(p):
        return pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p)
    sample = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    return pd.read_csv(sample) if os.path.exists(sample) else None

@app.route('/')
def index(): return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    df = load_df()
    stats = get_dashboard_stats(df) if df is not None else {}
    return render_template('dashboard.html', stats=stats,
                           has_data=bool(session.get('data_path')),
                           active_page='dashboard')

@app.route('/upload', methods=['GET','POST'])
def upload():
    msg=error=None; preview=[]; columns=[]; row_count=col_count=0
    if request.method == 'POST':
        f = request.files.get('file')
        if not f or f.filename == '':
            error = 'No file selected.'
        elif not allowed(f.filename):
            error = 'Invalid file type. Use CSV or Excel (.xlsx/.xls).'
        else:
            try:
                fn = secure_filename(f.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
                f.save(path); session['data_path'] = path
                df = pd.read_csv(path) if fn.endswith('.csv') else pd.read_excel(path)
                row_count, col_count = len(df), len(df.columns)
                columns = list(df.columns)
                preview = df.head(8).fillna('').to_dict('records')
                msg = f'Dataset uploaded — {row_count:,} rows × {col_count} columns detected.'
            except Exception as e:
                error = f'Error reading file: {e}'
    return render_template('upload.html', msg=msg, error=error, preview=preview,
                           columns=columns, row_count=row_count, col_count=col_count,
                           active_page='upload')

@app.route('/segmentation')
def segmentation():
    df = load_df()
    if df is None:
        return render_template('segmentation.html', error='No dataset loaded.',
                               active_page='segmentation', result=None)
    try:
        result = run_segmentation(df)
        return render_template('segmentation.html', result=result,
                               error=None, active_page='segmentation')
    except Exception as e:
        traceback.print_exc()
        return render_template('segmentation.html', error=str(e),
                               active_page='segmentation', result=None)

@app.route('/churn')
def churn():
    df = load_df()
    if df is None:
        return render_template('churn.html', error='No dataset loaded.',
                               active_page='churn', result=None)
    try:
        result = run_churn_prediction(df)
        return render_template('churn.html', result=result,
                               error=None, active_page='churn')
    except Exception as e:
        traceback.print_exc()
        return render_template('churn.html', error=str(e),
                               active_page='churn', result=None)

@app.route('/reports')
def reports():
    df = load_df()
    stats = get_dashboard_stats(df) if df is not None else {}
    return render_template('reports.html', stats=stats,
                           has_data=bool(session.get('data_path')),
                           active_page='reports')

@app.route('/api/use-sample')
def use_sample():
    session.pop('data_path', None)
    return redirect(url_for('dashboard'))

@app.route('/api/stats')
def api_stats():
    df = load_df()
    return jsonify(get_dashboard_stats(df) if df is not None else {})

if __name__ == '__main__':
    print("\n TeleInsight v2 starting...")
    print(" Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
