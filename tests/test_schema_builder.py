import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema_builder import build_structured_report, parse_cio_points_from_report, parse_market_indices


def test_parse_cio_points():
    text = """### Daily Point
_ Dow Jones 53,839.99 (-0.08%)
_ S&P 500 7,798.99 (+1.15%)

**Topline Signals**

- **미국 거시경제**: 7월 PPI 보합.<br />- **엔비디아**: 차세대 블랙웰 칩 가동.
"""
    points = parse_cio_points_from_report(text)
    assert len(points) == 2
    assert "미국 거시경제" in points[0]
    assert "엔비디아" in points[1]


def test_parse_market_indices():
    text = """### Daily Point
_ Dow Jones 53,839.99 (-0.08%)
_ S&P 500 7,798.99 (+1.15%)
_ Bitcoin 62,722.65 (-1.86%)
"""
    indices = parse_market_indices(text)
    assert "Dow Jones" in indices
    assert "53,839.99 (-0.08%)" == indices["Dow Jones"]
    assert "S&P 500" in indices
    assert "Bitcoin" in indices


def test_build_structured_report_flow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_date = "20260814"

        # Mock translated_state_20260814.json
        trans_data = {
            "https://test.com/nvda": {
                "title": "엔비디아 블랙웰 양산 돌입",
                "body": "Nvidia와 TSMC가 차세대 AI 칩을 양산합니다."
            }
        }
        with open(os.path.join(tmp_dir, f"translated_state_{target_date}.json"), "w", encoding="utf-8") as f:
            json.dump(trans_data, f)

        # Mock Semiconductor_sorted_20260814.json
        cat_data = [
            {
                "url": "https://test.com/nvda",
                "title": "Nvidia accelerates Blackwell",
                "content": "Nvidia and TSMC are scaling chip production."
            }
        ]
        with open(os.path.join(tmp_dir, f"Semiconductor_sorted_{target_date}.json"), "w", encoding="utf-8") as f:
            json.dump(cat_data, f)

        # Mock final_report_ko_20260814.txt
        with open(os.path.join(tmp_dir, f"final_report_ko_{target_date}.txt"), "w", encoding="utf-8") as f:
            f.write("**Topline Signals**\n\n- **반도체**: 엔비디아 상승세 지속.")

        out_path = build_structured_report(report_type="full", target_date=target_date, data_dir=tmp_dir)

        assert out_path is not None
        assert os.path.exists(out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        assert res["date"] == "2026-08-14"
        assert res["type"] == "full"
        assert len(res["cio_points"]) == 1
        assert len(res["articles"]) == 1
        art = res["articles"][0]
        assert art["category"] == "Semiconductor"
        assert "NVDA" in art["tickers"]
        assert "TSM" in art["tickers"]
        assert art["title_ko"] == "엔비디아 블랙웰 양산 돌입"
        assert art["content_ko"] == "Nvidia와 TSMC가 차세대 AI 칩을 양산합니다."
