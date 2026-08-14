"""
Ticker Matcher module for Alpha Signals Core.

Extracts stock tickers and cryptocurrency symbols from article text using an
Aho-Corasick automaton combined with token boundary validation and alias dictionary mapping.
Requires zero external LLM API calls (0 cost, ultra-fast matching).
"""

from collections import deque
import re
from typing import Dict, List, Set


class AhoCorasick:
    """Pure-Python Aho-Corasick automaton for high-throughput multi-pattern matching."""

    def __init__(self):
        self.trie: List[Dict[str, int]] = [{}]
        self.output: List[List[str]] = [[]]
        self.fail: List[int] = [0]
        self.word_map: Dict[str, str] = {}  # keyword -> canonical ticker

    def add_keyword(self, keyword: str, canonical_ticker: str):
        """Adds a keyword mapped to a canonical ticker symbol."""
        if not keyword:
            return
        curr = 0
        norm_kw = keyword.lower()
        self.word_map[norm_kw] = canonical_ticker
        for char in norm_kw:
            if char not in self.trie[curr]:
                self.trie[curr][char] = len(self.trie)
                self.trie.append({})
                self.output.append([])
                self.fail.append(0)
            curr = self.trie[curr][char]
        if canonical_ticker not in self.output[curr]:
            self.output[curr].append(canonical_ticker)

    def build(self):
        """Constructs failure links for the automaton using BFS."""
        queue = deque()
        for char, next_node in self.trie[0].items():
            self.fail[next_node] = 0
            queue.append(next_node)

        while queue:
            curr = queue.popleft()
            for char, next_node in self.trie[curr].items():
                queue.append(next_node)
                f = self.fail[curr]
                while f > 0 and char not in self.trie[f]:
                    f = self.fail[f]
                if char in self.trie[f] and self.trie[f][char] != next_node:
                    self.fail[next_node] = self.trie[f][char]
                else:
                    self.fail[next_node] = 0
                # Merge outputs
                self.output[next_node].extend(self.output[self.fail[next_node]])


# Top US Equities, Cryptos, and Sector leaders dictionary mapping (Aliases -> Ticker)
DEFAULT_TICKER_MAP: Dict[str, List[str]] = {
    # Big Tech / Mag 7
    "NVDA": ["NVDA", "Nvidia", "엔비디아"],
    "AAPL": ["AAPL", "Apple", "애플"],
    "MSFT": ["MSFT", "Microsoft", "마이크로소프트"],
    "GOOGL": ["GOOGL", "GOOG", "Google", "Alphabet", "구글", "알파벳"],
    "AMZN": ["AMZN", "Amazon", "아마존"],
    "META": ["META", "Meta", "메타", "Instagram", "인스타그램"],
    "TSLA": ["TSLA", "Tesla", "테슬라"],
    # Semiconductors / Hardware
    "TSM": ["TSM", "TSMC", "대만적체전로"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "AVGO": ["AVGO", "Broadcom", "브로드컴"],
    "QCOM": ["QCOM", "Qualcomm", "퀄컴"],
    "INTC": ["INTC", "Intel", "인텔"],
    "MU": ["MU", "Micron", "마이크론"],
    "ARM": ["ARM", "Arm Holdings"],
    "ASML": ["ASML"],
    "AMAT": ["AMAT", "Applied Materials", "어플라이드 머티어리얼즈"],
    "LRCX": ["LRCX", "Lam Research", "램리서치"],
    "KLAC": ["KLAC", "KLA"],
    "MRVL": ["MRVL", "Marvell", "마벨"],
    "SMCI": ["SMCI", "Super Micro", "슈퍼마이크로컴퓨터"],
    "DELL": ["DELL", "Dell", "델"],
    "HPE": ["HPE", "Hewlett Packard Enterprise"],
    # Software / Cloud / AI
    "ORCL": ["ORCL", "Oracle", "오라클"],
    "CRM": ["CRM", "Salesforce", "세일즈포스"],
    "PLTR": ["PLTR", "Palantir", "팔란티어"],
    "SNOW": ["SNOW", "Snowflake", "스노우플레이크"],
    "NOW": ["NOW", "ServiceNow", "서비스나우"],
    "IBM": ["IBM"],
    "CRWD": ["CRWD", "CrowdStrike", "크라우드스트라이크"],
    "PANW": ["PANW", "Palo Alto Networks", "팔로알토"],
    "NET": ["NET", "Cloudflare", "클라우드플레어"],
    "DDOG": ["DDOG", "Datadog", "데이터독"],
    "MDB": ["MDB", "MongoDB", "몽고DB"],
    "APP": ["APP", "AppLovin", "앱러빈"],
    "ADBE": ["ADBE", "Adobe", "어도비"],
    "INTU": ["INTU", "Intuit", "인튜이트"],
    "SHOP": ["SHOP", "Shopify", "쇼피파이"],
    "UBER": ["UBER", "Uber", "우버"],
    "ABNB": ["ABNB", "Airbnb", "에어비앤비"],
    "BKNG": ["BKNG", "Booking", "부킹홀딩스", "부킹"],
    "RDDT": ["RDDT", "Reddit", "레딧"],
    # Crypto / FinTech
    "BTC": ["BTC", "Bitcoin", "비트코인"],
    "ETH": ["ETH", "Ethereum", "이더리움"],
    "SOL": ["SOL", "Solana", "솔라나"],
    "XRP": ["XRP", "Ripple", "리플"],
    "COIN": ["COIN", "Coinbase", "코인베이스"],
    "MSTR": ["MSTR", "MicroStrategy", "마이크로스트래티지"],
    "HOOD": ["HOOD", "Robinhood", "로빈후드"],
    "SQ": ["SQ", "Block", "블록", "스퀘어"],
    "PYPL": ["PYPL", "PayPal", "페이팔"],
    # Financials
    "JPM": ["JPM", "JPMorgan", "제이피모간", "JP모건"],
    "BAC": ["BAC", "Bank of America", "뱅크오브아메리카"],
    "WFC": ["WFC", "Wells Fargo", "웰스파고"],
    "C": ["Citigroup", "씨티그룹", "씨티"],
    "GS": ["GS", "Goldman Sachs", "골드만삭스"],
    "MS": ["MS", "Morgan Stanley", "모건스탠리"],
    "V": ["Visa", "비자카드", "비자"],
    "MA": ["Mastercard", "마스터카드"],
    "AXP": ["AXP", "American Express", "아메리칸 익스프레스", "아멕스"],
    "BRK": ["Berkshire Hathaway", "버크셔 해서웨이", "버크셔"],
    # Energy / Power / Industrial
    "VRT": ["VRT", "Vertiv", "버티브"],
    "ETN": ["ETN", "Eaton", "이튼"],
    "CEG": ["CEG", "Constellation Energy", "콘스텔레이션 에너지"],
    "GEV": ["GEV", "GE Vernova", "GE버노바"],
    "NEE": ["NEE", "NextEra Energy", "넥스테라에너지"],
    "XOM": ["XOM", "ExxonMobil", "엑슨모빌"],
    "CVX": ["CVX", "Chevron", "셰브론"],
    "CAT": ["CAT", "Caterpillar", "캐터필러"],
    "DE": ["John Deere", "존디어", "디어"],
    "GE": ["General Electric", "제너럴 일렉트릭"],
    # Aerospace / Defense
    "LMT": ["LMT", "Lockheed Martin", "록히드마틴"],
    "RTX": ["RTX", "Raytheon"],
    "NOC": ["NOC", "Northrop Grumman", "노스롭그루먼"],
    "BA": ["BA", "Boeing", "보잉"],
    "RKLB": ["RKLB", "Rocket Lab", "로켓랩"],
    "SPCE": ["SPCE", "Virgin Galactic", "버진 갤럭틱"],
    # Healthcare / Bio
    "LLY": ["LLY", "Eli Lilly", "일라이 릴리", "일라이릴리"],
    "NVO": ["NVO", "Novo Nordisk", "노보 노디스크", "노보노디스크"],
    "PFE": ["PFE", "Pfizer", "화이자"],
    "MRK": ["MRK", "Merck", "머크"],
    "JNJ": ["JNJ", "Johnson & Johnson", "존슨앤존슨"],
    "UNH": ["UNH", "UnitedHealth", "유나이티드헬스"],
    "AMGN": ["AMGN", "Amgen", "암젠"],
    "GILD": ["GILD", "Gilead", "길리어드"],
    # Consumer / Retail
    "WMT": ["WMT", "Walmart", "월마트"],
    "COST": ["COST", "Costco", "코스트코"],
    "TGT": ["TGT", "Target", "타깃"],
    "NKE": ["NKE", "Nike", "나이키"],
    "SBUX": ["SBUX", "Starbucks", "스타벅스"],
    "MCD": ["MCD", "McDonald's", "맥도날드"],
    "DIS": ["DIS", "Disney", "디즈니"],
    "NFLX": ["NFLX", "Netflix", "넷플릭스"],
    "SPOT": ["SPOT", "Spotify", "스포티파이"],
}

# Ambiguous short tickers that must have strict boundary or $ prefix
AMBIGUOUS_TICKERS = {
    "A", "AI", "ON", "IT", "BE", "CAN", "ALL", "NOW", "SO", "CAT", "DE", "C", "V", "MS", "GS", "BA", "SQ", "GE"
}


class TickerMatcher:
    """High-speed Ticker Extraction engine."""

    def __init__(self, ticker_map: Dict[str, List[str]] = None):
        self.ticker_map = ticker_map or DEFAULT_TICKER_MAP
        self.automaton = AhoCorasick()
        self._init_automaton()

    def _init_automaton(self):
        for ticker, aliases in self.ticker_map.items():
            for alias in aliases:
                # Add exact alias
                self.automaton.add_keyword(alias, ticker)
        self.automaton.build()

    def extract_tickers(self, text: str) -> List[str]:
        """Extracts unique tickers from text using Aho-Corasick and regex token boundaries."""
        if not text:
            return []

        matched_tickers: Set[str] = set()

        # 1. Regex match for explicit $TICKER (e.g. $NVDA, $TSLA, $BTC)
        cashtags = re.findall(r"\$([A-Z]{1,5})\b", text)
        for tag in cashtags:
            if tag in self.ticker_map or len(tag) >= 2:
                matched_tickers.add(tag)

        # 2. Tokenize text into words/boundaries for ambiguity checks
        # Lowercase search with Aho-Corasick
        text_lower = text.lower()
        curr = 0
        for i, char in enumerate(text_lower):
            while curr > 0 and char not in self.automaton.trie[curr]:
                curr = self.automaton.fail[curr]
            if char in self.automaton.trie[curr]:
                curr = self.automaton.trie[curr][char]
            else:
                curr = 0

            for ticker in self.automaton.output[curr]:
                # If ambiguous ticker, check word boundary in original text
                if ticker in AMBIGUOUS_TICKERS:
                    # Check if exact uppercase standalone word exists or $ prefix
                    pattern = rf"(?:\$|\b){re.escape(ticker)}\b"
                    # Also check Korean alias match
                    aliases = self.ticker_map.get(ticker, [])
                    ko_aliases = [a for a in aliases if re.search(r"[\uac00-\ud7a3]", a)]
                    has_ko_match = any(ko in text for ko in ko_aliases)
                    if re.search(pattern, text) or has_ko_match:
                        matched_tickers.add(ticker)
                else:
                    matched_tickers.add(ticker)

        return sorted(list(matched_tickers))


# Singleton instance for quick usage
_matcher_instance = None


def get_ticker_matcher() -> TickerMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = TickerMatcher()
    return _matcher_instance


def extract_tickers_from_text(text: str) -> List[str]:
    """Convenience function to extract tickers from text."""
    matcher = get_ticker_matcher()
    return matcher.extract_tickers(text)
