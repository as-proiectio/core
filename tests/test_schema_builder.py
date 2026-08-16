import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema_builder import (
    build_structured_report,
    parse_markdown_articles,
)


def test_parse_markdown_articles():
    md = """### Daily Point
_ Dow Jones 50,000

### Semiconductor

[Nvidia surges](https://example.com/nvda)<br />
Nvidia Blackwell chips are in high demand.

### Software

[Microsoft expands Azure](https://example.com/msft)
Azure revenue crosses 100B.
"""
    arts = parse_markdown_articles(md)
    assert len(arts) == 2
    assert "https://example.com/nvda" in arts
    assert arts["https://example.com/nvda"]["category"] == "Semiconductor"
    assert arts["https://example.com/nvda"]["title"] == "Nvidia surges"
    assert (
        arts["https://example.com/nvda"]["content"]
        == "Nvidia Blackwell chips are in high demand."
    )


def test_build_structured_report_flow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_date = "20260814"

        # Mock report directory
        report_dir = os.path.join(tmp_dir, "report")
        os.makedirs(report_dir, exist_ok=True)

        en_md = """### Daily Point
_ S&P 500 7,700 (+1.0%)

### Semiconductor

[Nvidia Blackwell Ramp](https://test.com/nvda)<br />
Nvidia and TSMC are scaling chip production.
"""
        ko_md = """### Daily Point
_ S&P 500 7,700 (+1.0%)

### 반도체

[엔비디아 블랙웰 양산 돌입](https://test.com/nvda)<br />
Nvidia와 TSMC가 차세대 AI 칩을 양산합니다.
"""
        with open(
            os.path.join(report_dir, f"alpha_signal_{target_date}.md"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(en_md)

        with open(
            os.path.join(report_dir, f"alpha_signal_{target_date}_ko.md"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(ko_md)

        out_path = build_structured_report(
            report_type="full", target_date=target_date, data_dir=tmp_dir
        )

        assert out_path is not None
        assert os.path.exists(out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        assert res["date"] == "2026-08-14"
        assert res["type"] == "full"
        assert res["total_articles"] == 1
        assert len(res["articles"]) == 1
        art = res["articles"][0]
        assert art["category"] == "Semiconductor"
        assert "NVDA" in art["tickers"]
        assert "TSM" in art["tickers"]
        assert art["title"] == "Nvidia Blackwell Ramp"
        assert art["title_ko"] == "엔비디아 블랙웰 양산 돌입"
        assert art["content"] == "Nvidia and TSMC are scaling chip production."
        assert art["content_ko"] == "Nvidia와 TSMC가 차세대 AI 칩을 양산합니다."
