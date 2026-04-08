"""Tests rapides pour verifier les endpoints Flask."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Ajoute la racine du projet pour pouvoir importer api.app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import app


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("regression_loaded", data)
        self.assertIn("cluster_loaded", data)

    def test_predict_regression(self):
        payload = {
            "Date": "2023-01-01",
            "Qty": 2,
            "Category": "A",
            "Sales Channel": "Online",
            "Status": "Shipped",
        }
        resp = self.client.post("/predict_regression", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 1)

    def test_predict_cluster(self):
        payload = {
            "Date": "2023-01-01",
            "Qty": 2,
            "Category": "A",
            "Sales Channel": "Online",
            "Status": "Shipped",
        }
        resp = self.client.post("/predict_cluster", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("clusters", data)
        self.assertEqual(len(data["clusters"]), 1)


if __name__ == "__main__":
    unittest.main()
