import requests

HA_URL = "http://127.0.0.1:8123"

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1YWQ1Yjk5YmNiZmE0M2NiYjYwMTI1NTNjMThkNTZlZiIsImlhdCI6MTc5MTQxMDA4OCwiZXhwIjoyMTA2NzcwMDg4fQ.-5sP2evDlol_o94zzOQv1D-oseWFy_s9X93CtrnyJZI"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}


def run_script(entity_id):
    try:
        response = requests.post(
            f"{HA_URL}/api/services/script/turn_on",
            headers=HEADERS,
            json={"entity_id": entity_id},
            timeout=5
        )

        response.raise_for_status()
        print(f"SUCCESS: {entity_id}")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: {e}")


def bedroom_lights_on():
    run_script("script.kbedroomon")


def bedroom_lights_off():
    run_script("script.kbedroomoff")


# -------------------------
# TEST
# -------------------------

bedroom_lights_on()