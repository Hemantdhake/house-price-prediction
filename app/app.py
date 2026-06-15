# """
# Flask Web Application
# =====================
# Provides a web interface for house price predictions.
# """

# import logging
# import os
# from functools import wraps

# from flask import (
#     Flask,
#     render_template,
#     request,
#     jsonify,
#     redirect,
#     url_for,
# )

# # ─── App Setup ────────────────────────────────────────────────
# app = Flask(__name__)
# app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
# app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "False") == "True"

# # ─── Logger ───────────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ─── Load Predictor ───────────────────────────────────────────
# try:
#     import sys
#     sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
#     from src.predict import HousePricePredictor
#     predictor = HousePricePredictor()
#     MODEL_LOADED = True
#     logger.info("✅ Model loaded for Flask app.")
# except Exception as e:
#     logger.warning(f"⚠️ Could not load model: {e}")
#     predictor = None
#     MODEL_LOADED = False


# # ─── Error Handler ────────────────────────────────────────────
# def require_model(f):
#     """Decorator to check if model is loaded."""
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         if not MODEL_LOADED:
#             return jsonify({
#                 "error": "Model not loaded. Please train the model first.",
#                 "status": "error"
#             }), 503
#         return f(*args, **kwargs)
#     return decorated


# # ─── Routes ───────────────────────────────────────────────────

# @app.route("/")
# def index():
#     """Home page with prediction form."""
#     return render_template("index.html", model_loaded=MODEL_LOADED)


# @app.route("/predict", methods=["POST"])
# @require_model
# def predict():
#     """Handle prediction form submission."""
#     try:
#         # ── Parse Form Data ─────────────────────────────────
#         input_data = {
#             "LotArea": float(request.form.get("lot_area", 8500)),
#             "YearBuilt": int(request.form.get("year_built", 2000)),
#             "YearRemodAdd": int(request.form.get("year_remod", 2000)),
#             "TotalBsmtSF": float(request.form.get("basement_sf", 800)),
#             "GrLivArea": float(request.form.get("living_area", 1500)),
#             "FullBath": int(request.form.get("full_bath", 2)),
#             "HalfBath": int(request.form.get("half_bath", 0)),
#             "BedroomAbvGr": int(request.form.get("bedrooms", 3)),
#             "TotRmsAbvGrd": int(request.form.get("total_rooms", 6)),
#             "GarageArea": float(request.form.get("garage_area", 400)),
#             "PoolArea": float(request.form.get("pool_area", 0)),
#             "OverallQual": int(request.form.get("overall_qual", 5)),
#             "OverallCond": int(request.form.get("overall_cond", 5)),
#         }

#         # ── Make Prediction ──────────────────────────────────
#         result = predictor.predict_single(input_data)

#         return render_template(
#             "index.html",
#             prediction=result,
#             input_data=input_data,
#             model_loaded=MODEL_LOADED,
#         )

#     except Exception as e:
#         logger.error(f"❌ Prediction error: {e}")
#         return render_template(
#             "index.html",
#             error=str(e),
#             model_loaded=MODEL_LOADED,
#         )


# @app.route("/api/predict", methods=["POST"])
# @require_model
# def api_predict():
#     """REST API endpoint for predictions."""
#     try:
#         data = request.get_json(force=True)

#         if not data:
#             return jsonify({"error": "No data provided"}), 400

#         result = predictor.predict_single(data)

#         return jsonify({
#             "status": "success",
#             "prediction": result,
#         }), 200

#     except Exception as e:
#         logger.error(f"❌ API error: {e}")
#         return jsonify({"error": str(e), "status": "error"}), 500


# @app.route("/health")
# def health():
#     """Health check endpoint."""
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": MODEL_LOADED,
#         "version": "1.0.0",
#     }), 200


# @app.route("/api/model-info")
# def model_info():
#     """Return model information."""
#     return jsonify({
#         "model_name": "XGBoost Regressor",
#         "version": "1.0.0",
#         "target": "House Sale Price (USD)",
#         "features": [
#             "LotArea", "YearBuilt", "GrLivArea",
#             "OverallQual", "GarageArea", "FullBath",
#             "TotalBsmtSF", "BedroomAbvGr",
#         ],
#         "performance": {
#             "r2_score": 0.93,
#             "rmse": 22450,
#             "mae": 15680,
#         },
#     }), 200


# # ─── Error Handlers ───────────────────────────────────────────
# @app.errorhandler(404)
# def not_found(e):
#     return jsonify({"error": "Route not found", "status": 404}), 404


# @app.errorhandler(500)
# def server_error(e):
#     return jsonify({"error": "Internal server error", "status": 500}), 500


# # ─── Run App ──────────────────────────────────────────────────
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(
#         host="0.0.0.0",
#         port=port,
#         debug=app.config["DEBUG"],
#     )


# """
# Flask Web Application
# =====================
# Provides a web interface for house price predictions.
# """

# import logging
# import os
# from functools import wraps

# from flask import (
#     Flask,
#     render_template,
#     request,
#     jsonify,
#     redirect,
#     url_for,
# )

# # ─── App Setup ────────────────────────────────────────────────
# app = Flask(__name__)
# app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
# app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "False") == "True"

# # ─── Logger ───────────────────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ─── Load Predictor ───────────────────────────────────────────
# try:
#     import sys
#     sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
#     from src.predict import HousePricePredictor
#     predictor = HousePricePredictor()
#     MODEL_LOADED = True
#     logger.info("✅ Model loaded for Flask app.")
# except Exception as e:
#     logger.warning(f"⚠️ Could not load model: {e}")
#     predictor = None
#     MODEL_LOADED = False


# # ─── Error Handler ────────────────────────────────────────────
# def require_model(f):
#     """Decorator to check if model is loaded."""
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         if not MODEL_LOADED:
#             return jsonify({
#                 "error": "Model not loaded. Please train the model first.",
#                 "status": "error"
#             }), 503
#         return f(*args, **kwargs)
#     return decorated


# # ─── Routes ───────────────────────────────────────────────────

# @app.route("/")
# def index():
#     """Home page with prediction form."""
#     return render_template("index.html", model_loaded=MODEL_LOADED)


# @app.route("/predict", methods=["POST"])
# @require_model
# def predict():
#     """Handle prediction form submission."""
#     try:
#         # ── Parse Form Data ─────────────────────────────────
#         input_data = {
#             "LotArea": float(request.form.get("lot_area", 8500)),
#             "YearBuilt": int(request.form.get("year_built", 2000)),
#             "YearRemodAdd": int(request.form.get("year_remod", 2000)),
#             "TotalBsmtSF": float(request.form.get("basement_sf", 800)),
#             "GrLivArea": float(request.form.get("living_area", 1500)),
#             "FullBath": int(request.form.get("full_bath", 2)),
#             "HalfBath": int(request.form.get("half_bath", 0)),
#             "BedroomAbvGr": int(request.form.get("bedrooms", 3)),
#             "TotRmsAbvGrd": int(request.form.get("total_rooms", 6)),
#             "GarageArea": float(request.form.get("garage_area", 400)),
#             "PoolArea": float(request.form.get("pool_area", 0)),
#             "OverallQual": int(request.form.get("overall_qual", 5)),
#             "OverallCond": int(request.form.get("overall_cond", 5)),
#         }

#         # ── Make Prediction ──────────────────────────────────
#         result = predictor.predict_single(input_data)

#         return render_template(
#             "index.html",
#             prediction=result,
#             input_data=input_data,
#             model_loaded=MODEL_LOADED,
#         )

#     except Exception as e:
#         logger.error(f"❌ Prediction error: {e}")
#         return render_template(
#             "index.html",
#             error=str(e),
#             model_loaded=MODEL_LOADED,
#         )


# @app.route("/api/predict", methods=["POST"])
# @require_model
# def api_predict():
#     """REST API endpoint for predictions."""
#     try:
#         data = request.get_json(force=True)

#         if not data:
#             return jsonify({"error": "No data provided"}), 400

#         result = predictor.predict_single(data)

#         return jsonify({
#             "status": "success",
#             "prediction": result,
#         }), 200

#     except Exception as e:
#         logger.error(f"❌ API error: {e}")
#         return jsonify({"error": str(e), "status": "error"}), 500


# @app.route("/health")
# def health():
#     """Health check endpoint."""
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": MODEL_LOADED,
#         "version": "1.0.0",
#     }), 200


# @app.route("/api/model-info")
# def model_info():
#     """Return model information."""
#     return jsonify({
#         "model_name": "XGBoost Regressor",
#         "version": "1.0.0",
#         "target": "House Sale Price (USD)",
#         "features": [
#             "LotArea", "YearBuilt", "GrLivArea",
#             "OverallQual", "GarageArea", "FullBath",
#             "TotalBsmtSF", "BedroomAbvGr",
#         ],
#         "performance": {
#             "r2_score": 0.93,
#             "rmse": 22450,
#             "mae": 15680,
#         },
#     }), 200


# # ─── Error Handlers ───────────────────────────────────────────
# @app.errorhandler(404)
# def not_found(e):
#     return jsonify({"error": "Route not found", "status": 404}), 404


# @app.errorhandler(500)
# def server_error(e):
#     return jsonify({"error": "Internal server error", "status": 500}), 500


# # ─── Run App ──────────────────────────────────────────────────
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(
#         host="0.0.0.0",
#         port=port,
#         debug=app.config["DEBUG"],
#     )



"""
Streamlit Web Application
=========================
Provides a web interface for house price predictions.
"""

import logging
import os
import sys
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_predictor():
    """Load the model once and cache it across reruns."""
    try:
        # app.py lives at <project_root>/app/app.py
        # so project_root is two levels up from this file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.predict import HousePricePredictor
        predictor = HousePricePredictor()
        logger.info("✅ Model loaded for Streamlit app.")
        return predictor, True
    except Exception as e:
        logger.warning(f"⚠️ Could not load model: {e}")
        return None, False

predictor, MODEL_LOADED = load_predictor()

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .big-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/home.png", width=80)
    st.title("🏠 House Price Predictor")
    st.markdown("---")


    if MODEL_LOADED:
        st.success("✅ Model is loaded and ready", icon="✅")
    else:
        st.error("❌ Model not loaded — please train first", icon="❌")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Model Info", "📡 API Docs", "❤️ Health"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("v1.0.0  •  XGBoost Regressor")

if page == "🔮 Predict":

    if not MODEL_LOADED:
        st.warning("⚠️ Model is not loaded. Predictions are unavailable.")
        st.info("Train the model first, then restart the app.")
        st.stop()

    st.header("🔮 Predict House Price")
    st.markdown("Fill in the property details below, then hit **Predict**.")

    with st.form("prediction_form", clear_on_submit=False):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📐 Size & Area")
            lot_area = st.number_input(
                "Lot Area (sq ft)", min_value=500, max_value=500000,
                value=8500, step=100, help="Total lot size in square feet"
            )
            living_area = st.number_input(
                "Above-Grade Living Area (sq ft)", min_value=200, max_value=10000,
                value=1500, step=50, help="GrLivArea"
            )
            basement_sf = st.number_input(
                "Total Basement Area (sq ft)", min_value=0, max_value=10000,
                value=800, step=50, help="TotalBsmtSF"
            )
            garage_area = st.number_input(
                "Garage Area (sq ft)", min_value=0, max_value=5000,
                value=400, step=50, help="GarageArea"
            )
            pool_area = st.number_input(
                "Pool Area (sq ft)", min_value=0, max_value=2000,
                value=0, step=50, help="PoolArea"
            )

        with col2:
            st.subheader("🛏️ Rooms & Bathrooms")
            full_bath = st.number_input(
                "Full Bathrooms", min_value=0, max_value=5,
                value=2, step=1
            )
            half_bath = st.number_input(
                "Half Bathrooms", min_value=0, max_value=3,
                value=0, step=1
            )
            bedrooms = st.number_input(
                "Bedrooms Above Grade", min_value=0, max_value=10,
                value=3, step=1, help="BedroomAbvGr"
            )
            total_rooms = st.number_input(
                "Total Rooms Above Grade", min_value=1, max_value=15,
                value=6, step=1, help="TotRmsAbvGrd"
            )

        with col3:
            st.subheader("📅 Year & Quality")
            year_built = st.number_input(
                "Year Built", min_value=1800, max_value=2025,
                value=2000, step=1
            )
            year_remod = st.number_input(
                "Year Remodelled", min_value=1800, max_value=2025,
                value=2000, step=1, help="YearRemodAdd"
            )
            overall_qual = st.slider(
                "Overall Quality", min_value=1, max_value=10,
                value=5, step=1, help="OverallQual (1=Very Poor, 10=Excellent)"
            )
            overall_cond = st.slider(
                "Overall Condition", min_value=1, max_value=10,
                value=5, step=1, help="OverallCond (1=Very Poor, 10=Excellent)"
            )

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)

    if submitted:
        input_data = {
            "LotArea": float(lot_area),
            "YearBuilt": int(year_built),
            "YearRemodAdd": int(year_remod),
            "TotalBsmtSF": float(basement_sf),
            "GrLivArea": float(living_area),
            "FullBath": int(full_bath),
            "HalfBath": int(half_bath),
            "BedroomAbvGr": int(bedrooms),
            "TotRmsAbvGrd": int(total_rooms),
            "GarageArea": float(garage_area),
            "PoolArea": float(pool_area),
            "OverallQual": int(overall_qual),
            "OverallCond": int(overall_cond),
        }

        with st.spinner("Running prediction…"):
            try:
                result = predictor.predict_single(input_data)

                st.markdown("---")
                st.header("💰 Prediction Result")

                price = result.get("predicted_price", result.get("prediction", 0))

                # Big headline metric
                st.markdown(
                    f'<div class="metric-card">'
                    f'<p style="font-size:1rem;color:#666;margin-bottom:0;">Estimated Sale Price</p>'
                    f'<p class="big-number">${price:,.0f}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Confidence / range if available
                lower = result.get("lower_bound")
                upper = result.get("upper_bound")
                if lower and upper:
                    st.info(
                        f"📊 Predicted range: **${lower:,.0f}** — **${upper:,.0f}**"
                    )

                # Input summary
                with st.expander("📝 Input Summary", expanded=False):
                    st.json(input_data)

                # Full result detail
                with st.expander("🔍 Full Result Detail", expanded=False):
                    st.json(result)

                st.balloons()

            except Exception as e:
                logger.error(f"❌ Prediction error: {e}")
                st.error(f"❌ Prediction failed: {e}")

elif page == "📊 Model Info":

    st.header("📊 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ Architecture")
        st.markdown("""
        | Property       | Value                |
        |:--------------|:---------------------|
        | Model         | XGBoost Regressor    |
        | Version       | 1.0.0                |
        | Target        | House Sale Price (USD)|
        | Framework     | XGBoost + scikit-learn|
        """)

        st.subheader("🔑 Key Features")
        features = [
            "LotArea", "YearBuilt", "GrLivArea",
            "OverallQual", "GarageArea", "FullBath",
            "TotalBsmtSF", "BedroomAbvGr",
        ]
        for feat in features:
            st.markdown(f"- `{feat}`")

    with col2:
        st.subheader("📈 Performance Metrics")

        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", "0.93", delta=None)
        m2.metric("RMSE", "$22,450", delta=None)
        m3.metric("MAE", "$15,680", delta=None)

        st.markdown("---")
        st.subheader("📖 Feature Importance")

        # Simulated feature importance (replace with real data if available)
        importance_data = {
            "OverallQual": 0.28,
            "GrLivArea": 0.22,
            "GarageArea": 0.12,
            "TotalBsmtSF": 0.10,
            "YearBuilt": 0.09,
            "LotArea": 0.07,
            "FullBath": 0.06,
            "BedroomAbvGr": 0.04,
        }

        try:
            import pandas as pd
            df_imp = pd.DataFrame(
                list(importance_data.items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=True)

            st.bar_chart(df_imp, x="Feature", y="Importance", height=350)
        except Exception:
            for feat, imp in sorted(importance_data.items(), key=lambda x: x[1]):
                st.progress(imp, text=f"{feat}: {imp:.0%}")

# ─── Page: API Docs ───────────────────────────────────────────
elif page == "📡 API Docs":

    st.header("📡 API Documentation")

    st.markdown("""
    The original Flask app exposed a REST API. Below is the reference if you
    still need to call the prediction service programmatically.
    """)

    st.subheader("POST `/api/predict`")
    st.markdown("Send a JSON payload with house features to get a prediction.")

    st.code("""
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "LotArea": 8500,
    "YearBuilt": 2000,
    "YearRemodAdd": 2000,
    "TotalBsmtSF": 800,
    "GrLivArea": 1500,
    "FullBath": 2,
    "HalfBath": 0,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 6,
    "GarageArea": 400,
    "PoolArea": 0,
    "OverallQual": 5,
    "OverallCond": 5
  }'
    """, language="bash")

    st.subheader("Expected Response")
    st.code("""
{
  "status": "success",
  "prediction": {
    "predicted_price": 185000.0,
    "lower_bound": 165000.0,
    "upper_bound": 205000.0
  }
}
    """, language="json")

    st.markdown("---")
    st.info("💡 **Tip:** In a Streamlit-only deployment, you can embed a FastAPI "
            "sidecar or use `st.experimental_api` if you need a REST endpoint.")

elif page == "❤️ Health":

    st.header("❤️ Health Check")

    col1, col2, col3 = st.columns(3)
    col1.metric("App Status", "Healthy ✅")
    col2.metric("Model Loaded", "Yes ✅" if MODEL_LOADED else "No ❌")
    col3.metric("Version", "1.0.0")

    st.markdown("---")
    st.subheader("Raw JSON (equivalent to `/health`)")

    health_payload = {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0",
    }
    st.json(health_payload)