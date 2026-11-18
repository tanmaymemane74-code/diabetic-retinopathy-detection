"""
Test live API endpoints against running FastAPI server (http://127.0.0.1:8000).
"""
import os
import requests

def test_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Health
    r = requests.get(f"{base_url}/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("[PASS] /api/health:", r.json().get("status"))
    
    # Find sample image
    sample_dirs = ["dataset/raw/train_images", "dataset/synthetic_samples"]
    img_path = None
    for s_dir in sample_dirs:
        if os.path.exists(s_dir):
            images = [f for f in os.listdir(s_dir) if f.endswith((".png", ".jpg"))]
            if images:
                img_path = os.path.join(s_dir, images[0])
                break
        
    print(f"Using image for tests: {img_path}")
    
    # 2. Predict
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(f"{base_url}/api/predict", files=files)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred = r.json()
    print("[PASS] /api/predict:", pred.get("predicted_class"), "Confidence:", pred.get("confidence"))
    
    # 3. Explain (Grad-CAM)
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(f"{base_url}/api/explain", files=files)
    assert r.status_code == 200, f"Explain failed: {r.text}"
    exp = r.json()
    print("[PASS] /api/explain: heatmap_b64 length:", len(exp.get("heatmap_base64", "")))
    
    # 4. Clinical Guidance
    grade = pred.get("predicted_grade", 2)
    r = requests.get(f"{base_url}/api/clinical-guidance/{grade}")
    assert r.status_code == 200, f"Guidance failed: {r.text}"
    guidance = r.json()
    print("[PASS] /api/clinical-guidance:", guidance.get("suggested_action", {}).get("action"))
    
    # 5. Full Screening Endpoint
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        data = {
            "patient_id": "PAT-TEST-002",
            "patient_name": "John Doe",
            "patient_age": 58,
            "eye": "left",
            "save_record": "true"
        }
        r = requests.post(f"{base_url}/api/screen", files=files, data=data)
    assert r.status_code == 200, f"Screen failed: {r.text}"
    screen = r.json()
    print("[PASS] /api/screen: Screening ID:", screen.get("screening_id"))
    
    print("\nALL LIVE API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
