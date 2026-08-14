import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ticker_matcher import TickerMatcher, extract_tickers_from_text


def test_basic_ticker_extraction():
    text = "엔비디아(Nvidia)는 블랙웰 AI 칩 양산에 돌입했으며, TSMC($TSM)와 협력 중입니다."
    tickers = extract_tickers_from_text(text)
    assert "NVDA" in tickers
    assert "TSM" in tickers


def test_cashtag_extraction():
    text = "Bitcoin is testing $BTC 65k while $SOL and $ETH show strong inflows."
    tickers = extract_tickers_from_text(text)
    assert "BTC" in tickers
    assert "SOL" in tickers
    assert "ETH" in tickers


def test_ambiguous_tickers_avoid_false_positives():
    # 'on', 'can', 'be', 'it' shouldn't falsely trigger unless context / uppercase match
    text = "It can be an on-going development in tech."
    tickers = extract_tickers_from_text(text)
    assert "CAN" not in tickers
    assert "BE" not in tickers
    assert "IT" not in tickers


def test_korean_alias_matching():
    text = "마이크로소프트와 아마존, 구글이 클라우드 인프라 투자를 대폭 확대했습니다."
    tickers = extract_tickers_from_text(text)
    assert "MSFT" in tickers
    assert "AMZN" in tickers
    assert "GOOGL" in tickers
