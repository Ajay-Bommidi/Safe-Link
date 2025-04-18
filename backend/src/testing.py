import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
import requests
from urllib.parse import urlparse, quote
import argparse
from datetime import datetime
import tldextract
import re
from sklearn.base import is_classifier
from sklearn.metrics import f1_score

# Setup logging with UTF-8 encoding
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("url_prediction.log", encoding='utf-8'), 
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Config
MODEL_DIR = "../models/"
DATA_DIR = "../data/final"
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_model_optimized.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback.csv")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "optimal_threshold.pkl")

# API Keys (Replace with secure environment variables in production)
VIRUSTOTAL_API_KEY = "API KEY"
GOOGLE_SAFE_BROWSING_API_KEY = "API KEY"  # Verify this key is correct
VIRUS_TOTAL_URL = "https://www.virustotal.com/api/v3/urls"
VIRUS_TOTAL_ANALYSIS_URL = "https://www.virustotal.com/api/v3/analyses"
GOOGLE_SAFE_BROWSING_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
TIMEOUT = 5  # Reduced timeout for faster responses
MAX_RETRIES = 3  # Reduced retries to minimize delay

# Feature extraction constants
EXPECTED_FEATURE_COUNT = 167
KNOWN_LEGIT_DOMAINS = {
    'facebook.com', 'google.com', 'amazon.com', 'twitter.com', 'youtube.com', 'linkedin.com',
    'microsoft.com', 'apple.com', 'bankofamerica.com', 'chase.com', 'wellsfargo.com', 'citibank.com',
    'paypal.com', 'news.ycombinator.com', 'github.io', 'thehindu.com', 'sircrrengg.ac.in', 'flipkart.com',
    'amazon.in', 'netflix.com', 'jiohotstar.com', 'instagram.com', 'zendesk.com', 'twitch.tv',
    'meesho.com', 'myntra.com', 'lenskart.com', 'zomato.com','blogger.com'
}
SUSPICIOUS_KEYWORDS = {
    'login', 'secure', 'account', 'verify', 'update', 'payment', 'bank', 'credit', 'password', 
    'confirm', 'reset', 'alert', 'urgent', 'signin', 'fraud', 'hack', 'steal', 'access', 'credential'
}
SUSPICIOUS_PATTERNS = [
    r'\d{3,}', r'[\-_]{3,}', r'tmp|test|demo|random', r'@\w+', r'\.php|\.asp|\.jsp|\.js',
    r'[^a-zA-Z0-9\.\-\/:\?\=&-]{2,}', r'user|pass|token|key|id|secret', r'http.*http'
]

def extract_features(url):
    """Extract features from a URL with named features for compatibility with LGBMClassifier."""
    try:
        parsed_url = urlparse(url)
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}" if extracted.domain and extracted.suffix else parsed_url.netloc
        path = parsed_url.path or "/"
        query = parsed_url.query or ""
        url_lower = url.lower()

        # Define feature names explicitly
        feature_names = [f"feat_{i}" for i in range(EXPECTED_FEATURE_COUNT)]
        features = {
            'feat_0_length_url': len(url),
            'feat_1_length_hostname': len(domain),
            'feat_2_nb_dots': url.count('.'),
            'feat_3_nb_hyphens': url.count('-'),
            'feat_4_nb_at': url.count('@'),
            'feat_5_nb_qm': url.count('?'),
            'feat_6_nb_and': url.count('&'),
            'feat_7_nb_eq': url.count('='),
            'feat_8_nb_slash': url.count('/'),
            'feat_9_nb_www': 1 if 'www' in domain.lower() else 0,
            'feat_10_nb_com': 1 if '.com' in domain.lower() else 0,
            'feat_11_https_token': 1 if url.startswith('https') else 0,
            'feat_12_ratio_digits_url': sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
            'feat_13_ratio_digits_host': sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0,
            'feat_14_nb_subdomains': len(extracted.subdomain.split('.')) if extracted.subdomain else 0,
            'feat_15_phish_hints': sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower),
            'feat_16_suspecious_tld': 1 if extracted.suffix not in {'com', 'org', 'net', 'edu', 'gov', 'co', 'io', 'ac.in'} else 0,
            'feat_17_is_known_legit': 1 if domain in KNOWN_LEGIT_DOMAINS else 0,
            'feat_18_has_suspicious_pattern': 1 if any(re.search(p, url_lower) for p in SUSPICIOUS_PATTERNS) else 0,
            'feat_19_domain_length': len(domain),
            'feat_20_path_length': len(path),
            'feat_21_query_length': len(query),
        }

        # Pad remaining features with zeros
        feature_vector = np.zeros(EXPECTED_FEATURE_COUNT)
        for i, name in enumerate(feature_names):
            feature_vector[i] = features.get(name, 0)

        # Return as DataFrame with feature names for LGBM compatibility
        return pd.DataFrame(feature_vector.reshape(1, -1), columns=feature_names)
    except Exception as e:
        logger.error(f"Error extracting features for {url}: {str(e)}")
        return pd.DataFrame(np.zeros((1, EXPECTED_FEATURE_COUNT)), columns=[f"feat_{i}" for i in range(EXPECTED_FEATURE_COUNT)])

def check_virus_total(url, retries=MAX_RETRIES):
    """Check URL with VirusTotal API with retries and fallback."""
    try:
        headers = {"x-apikey": VIRUSTOTAL_API_KEY}
        payload = {"url": quote(url, safe=':/')}
        for attempt in range(retries):
            try:
                response = requests.post(VIRUS_TOTAL_URL, headers=headers, data=payload, timeout=TIMEOUT)
                response.raise_for_status()
                scan_id = response.json()['data']['id']
                analysis_url = f"{VIRUS_TOTAL_ANALYSIS_URL}/{scan_id}"
                for _ in range(5):  # Reduced polling attempts
                    analysis_response = requests.get(analysis_url, headers=headers, timeout=TIMEOUT)
                    analysis_response.raise_for_status()
                    analysis = analysis_response.json()
                    if analysis['data']['attributes']['status'] == 'completed':
                        stats = analysis['data']['attributes']['stats']
                        malicious = stats['malicious']
                        total = sum(stats.values())
                        confidence = malicious / total if total > 0 else 0.0
                        return 1 if malicious > 0 else 0, max(0.1, min(0.9, confidence))
                    time.sleep(1)
                logger.warning(f"VirusTotal analysis incomplete for {url}")
                return -1, 0.0
            except requests.RequestException as e:
                if attempt < retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{retries} for VirusTotal: {str(e)}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"VirusTotal failed after retries: {str(e)}")
                    return -1, 0.0
    except Exception as e:
        logger.error(f"Unexpected error in VirusTotal for {url}: {str(e)}")
        return -1, 0.0

def check_google_safe_browsing(url, retries=MAX_RETRIES):
    """Check URL with Google Safe Browsing API with retries and fallback."""
    try:
        payload = {
            "client": {"clientId": "safe-link-checker", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["SOCIAL_ENGINEERING", "MALWARE"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": quote(url, safe=':/')}]
            }
        }
        for attempt in range(retries):
            try:
                response = requests.post(GOOGLE_SAFE_BROWSING_URL, params={"key": GOOGLE_SAFE_BROWSING_API_KEY}, 
                                        json=payload, timeout=TIMEOUT)
                response.raise_for_status()
                result = response.json()
                is_malicious = 1 if "matches" in result else 0
                confidence = 0.9 if is_malicious else 0.1
                return is_malicious, confidence
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    logger.error(f"Google Safe Browsing 400 Bad Request for {url}: {str(e.response.text)}")
                    return -1, 0.0
                if attempt < retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{retries} for Google Safe Browsing: {str(e)}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Google Safe Browsing failed after retries: {str(e)}")
                    return -1, 0.0
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{retries} for Google Safe Browsing: {str(e)}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Unexpected error in Google Safe Browsing for {url}: {str(e)}")
                    return -1, 0.0
    except Exception as e:
        logger.error(f"Critical error in Google Safe Browsing for {url}: {str(e)}")
        return -1, 0.0

def load_model_and_threshold():
    """Load model, scaler, and optimal threshold with fallback."""
    try:
        logger.info("Loading model, scaler, and threshold...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        if os.path.exists(THRESHOLD_PATH):
            optimal_threshold = joblib.load(THRESHOLD_PATH)
        else:
            logger.warning("Optimal threshold not found, estimating threshold...")
            optimal_threshold = estimate_default_threshold(model, scaler)
        if not is_classifier(model):
            raise ValueError("Loaded model is not a classifier")
        return model, scaler, optimal_threshold
    except Exception as e:
        logger.error(f"Failed to load model or scaler: {str(e)}")
        raise

def estimate_default_threshold(model, scaler):
    """Estimate a default threshold using sample data."""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "fina.csv"), nrows=1000, dtype=np.float32)
        if 'url' in df.columns:
            df = df.drop(columns=['url'])
        if 'label' not in df.columns:
            logger.error("Label column missing in fina.csv")
            return 0.5
        X = df.drop(columns=['label']).values
        y = df['label'].values
        if X.shape[1] != EXPECTED_FEATURE_COUNT:
            logger.error(f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, got {X.shape[1]}")
            return 0.5

        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        X_values = X_df.values
        X_scaled = scaler.transform(X_values)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)
        y_pred_proba = model.predict_proba(X_scaled_df)[:, 1]
        return find_optimal_threshold(y_pred_proba, y, min_threshold=0.1, max_threshold=0.9, step=0.01)
    except Exception as e:
        logger.warning(f"Threshold estimation failed: {str(e)}, defaulting to 0.5")
        return 0.5

def find_optimal_threshold(y_pred_proba, y_true, min_threshold=0.1, max_threshold=0.9, step=0.01):
    """Find the optimal threshold for maximizing F1-score."""
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(min_threshold, max_threshold + step, step):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

def classify_url(model, scaler, url, optimal_threshold):
    """Classify a URL using model and APIs with robust feature handling and prediction logic."""
    logger.info(f"Processing URL: {url}")
    start_time = time.time()

    try:
        # Extract features as a DataFrame
        features_df = extract_features(url)
        if features_df.shape[1] != EXPECTED_FEATURE_COUNT:
            logger.warning(f"Feature mismatch for {url}, expected {EXPECTED_FEATURE_COUNT}, got {features_df.shape[1]}")
            features_df = pd.DataFrame(np.zeros((1, EXPECTED_FEATURE_COUNT)), 
                                     columns=[f"feat_{i}" for i in range(EXPECTED_FEATURE_COUNT)])

        # Scale features: NumPy array for scaler, then DataFrame for model
        features_values = features_df.values
        try:
            features_scaled = scaler.transform(features_values)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns)
        except Exception as e:
            logger.warning(f"Scaler transform failed for {url}: {str(e)}, using raw features")
            features_scaled_df = features_df  # Fallback

        # Model prediction
        try:
            model_pred = model.predict(features_scaled_df)[0]
            model_conf = model.predict_proba(features_scaled_df)[0][int(model_pred)]
        except Exception as e:
            logger.error(f"Model prediction failed for {url}: {str(e)}")
            model_pred, model_conf = 0, 0.5  # Neutral fallback

        # API checks
        vt_pred, vt_conf = check_virus_total(url)
        gsb_pred, gsb_conf = check_google_safe_browsing(url)

        # Combine predictions
        valid_preds = []
        valid_confs = []
        domain = tldextract.extract(url).registered_domain

        if domain in KNOWN_LEGIT_DOMAINS and not any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS):
            final_pred, final_conf, label = 0, 0.95, "Legitimate"
        else:
            if model_conf >= 0.6:  # Only include model if confidence is sufficient
                valid_preds.append(model_pred)
                valid_confs.append(model_conf * 0.6 if model_conf > 0.7 else model_conf * 0.3)
            if vt_pred != -1:
                valid_preds.append(vt_pred)
                valid_confs.append(vt_conf * 0.35)
            if gsb_pred != -1:
                valid_preds.append(gsb_pred)
                valid_confs.append(gsb_conf * 0.35)

            if not valid_preds:
                logger.warning(f"No valid predictions for {url}, using fallback")
                final_pred, final_conf = 0, 0.5 if domain in KNOWN_LEGIT_DOMAINS else 0.5
                label = "Legitimate" if final_pred == 0 else "Phishing"
            else:
                weighted_sum = sum(p * c for p, c in zip(valid_preds, valid_confs))
                total_conf = sum(valid_confs)
                final_prob = weighted_sum / total_conf
                final_pred = 1 if final_prob >= optimal_threshold else 0
                final_conf = min(1.0, max(0.8, final_prob if final_pred else 1 - final_prob))
                label = "Phishing" if final_pred else "Legitimate"

        logger.info(f"URL: {url}, Prediction: {label}, Confidence: {final_conf:.4f}, "
                    f"Model: {model_pred} ({model_conf:.4f}), VT: {vt_pred} ({vt_conf:.4f}), GSB: {gsb_pred} ({gsb_conf:.4f})")

        result = {
            "url": url, "model_pred": int(model_pred), "model_conf": float(model_conf),
            "vt_pred": int(vt_pred) if vt_pred != -1 else "N/A", "vt_conf": float(vt_conf) if vt_pred != -1 else "N/A",
            "gsb_pred": int(gsb_pred) if gsb_pred != -1 else "N/A", "gsb_conf": float(gsb_conf) if gsb_pred != -1 else "N/A",
            "final_pred": int(final_pred), "final_conf": float(final_conf), "label": label,
            "timestamp": datetime.now().isoformat(), "feedback": None
        }
        return result, time.time() - start_time
    except Exception as e:
        logger.error(f"Critical error classifying {url}: {str(e)}")
        domain = tldextract.extract(url).registered_domain
        vt_pred, vt_conf = check_virus_total(url)
        gsb_pred, gsb_conf = check_google_safe_browsing(url)
        final_pred = 1 if (vt_pred == 1 or gsb_pred == 1) else 0
        final_conf = max(vt_conf, gsb_conf, 0.5) if final_pred else 0.5
        label = "Phishing" if final_pred else "Legitimate"
        result = {
            "url": url, "model_pred": -1, "model_conf": 0.0, "vt_pred": int(vt_pred) if vt_pred != -1 else "N/A",
            "vt_conf": float(vt_conf) if vt_pred != -1 else "N/A", "gsb_pred": int(gsb_pred) if gsb_pred != -1 else "N/A",
            "gsb_conf": float(gsb_conf) if gsb_pred != -1 else "N/A", "final_pred": int(final_pred),
            "final_conf": float(final_conf), "label": label, "timestamp": datetime.now().isoformat(), "feedback": None
        }
        return result, time.time() - start_time

def save_feedback(results):
    """Save predictions and feedback to CSV with deduplication."""
    try:
        df = pd.DataFrame(results)
        if os.path.exists(FEEDBACK_CSV):
            existing_df = pd.read_csv(FEEDBACK_CSV)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=["url", "timestamp"], keep="last")
        df.to_csv(FEEDBACK_CSV, index=False)
        logger.info(f"Saved {len(results)} predictions to '{FEEDBACK_CSV}'")
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")

def process_urls(input_source):
    """Process single URL or CSV file with error handling."""
    try:
        model, scaler, optimal_threshold = load_model_and_threshold()
        results = []
        if os.path.isfile(input_source):
            df = pd.read_csv(input_source)
            if 'url' not in df.columns:
                raise ValueError("CSV must contain 'url' column")
            urls = df['url'].tolist()
        else:
            urls = [input_source]

        for url in urls:
            result, latency = classify_url(model, scaler, url, optimal_threshold)
            results.append(result)
            logger.info(f"Processed {url} in {latency:.2f}s")
        save_feedback(results)
        return results
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return []

def retrain_model():
    """Retrain the model using feedback.csv data."""
    try:
        if not os.path.exists(FEEDBACK_CSV):
            logger.warning("No feedback data available for retraining")
            return

        feedback_df = pd.read_csv(FEEDBACK_CSV)
        if "final_pred" not in feedback_df.columns or "feedback" not in feedback_df.columns:
            logger.error("Feedback CSV missing required columns")
            return

        y_feedback = feedback_df["feedback"].fillna(feedback_df["final_pred"]).astype(int)
        X_feedback = feedback_df.drop(columns=["url", "label", "timestamp", "feedback", "final_pred", 
                                             "model_pred", "model_conf", "vt_pred", "vt_conf", 
                                             "gsb_pred", "gsb_conf"]).values

        if len(X_feedback) == 0 or len(y_feedback) == 0:
            logger.warning("No valid feedback data for retraining")
            return

        X_feedback_df = pd.DataFrame(X_feedback, columns=[f"feat_{i}" for i in range(X_feedback.shape[1])])
        X_values = X_feedback_df.values
        X_scaled = scaler.transform(X_values)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_feedback_df.columns)

        model, scaler, _ = load_model_and_threshold()
        model.fit(X_scaled_df, y_feedback)
        y_pred_proba = model.predict_proba(X_scaled_df)[:, 1]
        optimal_threshold = find_optimal_threshold(y_pred_proba, y_feedback)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(optimal_threshold, THRESHOLD_PATH)
        logger.info(f"✅ Model retrained with feedback data and saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Retrain failed: {str(e)}")

def main():
    """Main execution for URL prediction or retraining."""
    parser = argparse.ArgumentParser(description="Predict phishing/legitimate URLs with model and APIs, or retrain model.")
    parser.add_argument("--input", type=str, help="Single URL or path to CSV file with 'url' column")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model using feedback.csv")
    args = parser.parse_args()

    start_time = time.time()
    try:
        if args.retrain:
            retrain_model()
        elif args.input:
            process_urls(args.input)
        else:
            raise ValueError("Specify --input for prediction or --retrain for model retraining")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
    finally:
        logger.info(f"Process completed in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
    
