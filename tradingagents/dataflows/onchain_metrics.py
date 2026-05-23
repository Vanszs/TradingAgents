"""On-chain metrics via Etherscan API.

Contract addresses are auto-resolved from CoinGecko's `platforms` field,
so any EVM token is supported without hardcoding. Requires ETHERSCAN_API_KEY.
Gracefully degrades without the key.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from .crypto_id_map import ticker_to_coingecko_id

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL = 300  # 5 minutes

# Fast-path: known contract addresses (ticker → contract or "native")
# Auto-discovery via CoinGecko fills this cache at runtime for unknown tokens.
_TOKEN_CONTRACTS: dict[str, tuple[str, str]] = {
    # ticker: (contract_address_or_"native", chain_name)
    "ETH-USD":  ("native", "ethereum"),
    "USDT-USD": ("0xdac17f958d2ee523a2206206994597c13d831ec7", "ethereum"),
    "USDC-USD": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "ethereum"),
    "LINK-USD": ("0x514910771af9ca656af840dff83e8264ecf986ca", "ethereum"),
    "UNI-USD":  ("0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", "ethereum"),
    "AAVE-USD": ("0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", "ethereum"),
    "MKR-USD":  ("0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2", "ethereum"),
    "CRV-USD":  ("0xd533a949740bb3306d119cc777fa900ba034cd52", "ethereum"),
    "SHIB-USD": ("0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce", "ethereum"),
    "PEPE-USD": ("0x6982508145454ce325ddbe47a25d4ec3d2311933", "ethereum"),
}

# Native L1 chains — no EVM contract, Etherscan not applicable
_NATIVE_L1 = {"BTC", "SOL", "XRP", "ADA", "DOT", "ATOM", "XLM", "ALGO",
              "NEAR", "APT", "SUI", "FIL", "TRX", "TON", "DOGE", "LTC"}

# Chain preference order for EVM contract resolution
_CHAIN_PREFERENCE = ["ethereum", "binance-smart-chain", "polygon-pos",
                     "arbitrum-one", "optimistic-ethereum", "base"]

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"


def _get_api_key() -> Optional[str]:
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    return key if key else None


def _resolve_contract_address(ticker: str) -> Optional[tuple[str, str]]:
    """Return (contract_address, chain) for a ticker, or None if not EVM.

    Fast path: hardcoded map.
    Slow path: CoinGecko `platforms` field (auto-discovery).
    Result is cached in _TOKEN_CONTRACTS for future calls.
    """
    normalized = ticker.strip().upper()

    # Fast path
    if normalized in _TOKEN_CONTRACTS:
        return _TOKEN_CONTRACTS[normalized]

    # Base symbol check (e.g. ETH-USDT → ETH-USD)
    base = normalized.split("-")[0]
    if base in _NATIVE_L1:
        return None  # Not an EVM token

    # Slow path: CoinGecko platforms
    coin_id = ticker_to_coingecko_id(ticker)
    if not coin_id:
        return None

    try:
        from .coingecko import _get
        data = _get(f"/coins/{coin_id}", params={
            "localization": "false", "tickers": "false",
            "market_data": "false", "community_data": "false",
            "developer_data": "false",
        })
        if not data:
            return None

        platforms: dict[str, str] = data.get("platforms", {})
        # Remove empty entries
        platforms = {k: v for k, v in platforms.items() if v}
        if not platforms:
            return None

        # Pick preferred chain
        contract, chain = None, None
        for preferred in _CHAIN_PREFERENCE:
            if preferred in platforms:
                contract, chain = platforms[preferred], preferred
                break
        if not contract:
            # Take first available
            chain, contract = next(iter(platforms.items()))

        result = (contract, chain)
        _TOKEN_CONTRACTS[normalized] = result  # cache for future calls
        return result

    except Exception as exc:
        logger.warning("CoinGecko platforms lookup failed for %s: %s", ticker, exc)
        return None


def _etherscan_get(params: dict, chainid: str = "1") -> Optional[dict]:
    key = _get_api_key()
    if not key:
        return None
    cache_key = str(sorted({**params, "chainid": chainid}.items()))
    now = time.time()
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            return data  # type: ignore
    try:
        full_params = {"chainid": chainid, "apikey": key, **params}
        resp = requests.get(ETHERSCAN_BASE, params=full_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1":
            _CACHE[cache_key] = (now, data)
            return data
        return None
    except Exception as exc:
        logger.warning("Etherscan request failed: %s", exc)
        return None


def get_onchain_metrics(ticker: str) -> str:
    """Return on-chain metrics for a crypto asset as a formatted string."""
    base = ticker.split("-")[0].upper()

    # Native L1 — Etherscan not applicable
    if base in _NATIVE_L1:
        return (
            f"## On-Chain Metrics: {ticker}\n\n"
            f"{base} is a native Layer-1 chain, not an EVM token. "
            "Etherscan on-chain metrics (active addresses, tx count) are not applicable. "
            "For BTC: use blockchain.info or mempool.space. "
            "For SOL: use solscan.io."
        )

    if not _get_api_key():
        return (
            f"## On-Chain Metrics: {ticker}\n\n"
            "ETHERSCAN_API_KEY not configured. "
            "Set this env var to enable on-chain analysis (active addresses, tx count, gas usage). "
            "Register free at https://etherscan.io/apis"
        )

    resolved = _resolve_contract_address(ticker)
    if not resolved:
        return (
            f"## On-Chain Metrics: {ticker}\n\n"
            f"Could not resolve contract address for {ticker}. "
            "This token may not be an EVM token or is not listed on CoinGecko."
        )

    contract, chain = resolved
    if contract == "native":
        return _get_eth_native_metrics(ticker)
    return _get_erc20_metrics(ticker, contract, chain)


def _get_eth_native_metrics(ticker: str) -> str:
    supply_data = _etherscan_get({"module": "stats", "action": "ethsupply2"})
    lines = [f"## On-Chain Metrics: {ticker} (Ethereum Native)"]
    if supply_data:
        result = supply_data.get("result", {})
        eth_supply = int(result.get("EthSupply", 0)) / 1e18
        burned = int(result.get("BurntFees", 0)) / 1e18
        lines += [
            "",
            f"- Total ETH Supply: {eth_supply:,.0f} ETH",
            f"- Total ETH Burned (EIP-1559): {burned:,.2f} ETH",
            f"- Net Issuance: {eth_supply - burned:,.0f} ETH (approx)",
        ]
    else:
        lines.append("\nSupply data unavailable.")
    return "\n".join(lines)


def _get_erc20_metrics(ticker: str, contract: str, chain: str = "ethereum") -> str:
    token_data = _etherscan_get({
        "module": "stats", "action": "tokensupply",
        "contractaddress": contract,
    })
    # Query token decimals (defaults to 18 if unavailable)
    decimals = 18
    dec_data = _etherscan_get({
        "module": "proxy", "action": "eth_call",
        "to": contract,
        "data": "0x313ce567",  # decimals() selector
        "tag": "latest",
    })
    if dec_data and dec_data.get("result"):
        try:
            decimals = int(dec_data["result"], 16)
        except (ValueError, TypeError):
            pass
    lines = [f"## On-Chain Metrics: {ticker} (Chain: {chain})"]
    if token_data:
        raw = token_data.get("result", "0")
        try:
            # Use queried decimals for correct human-readable supply
            supply_raw = int(raw)
            supply_human = supply_raw / (10 ** decimals)
            if supply_human >= 1e9:
                supply_fmt = f"{supply_human/1e9:.2f}B tokens"
            elif supply_human >= 1e6:
                supply_fmt = f"{supply_human/1e6:.2f}M tokens"
            else:
                supply_fmt = f"{supply_human:,.0f} tokens"
        except (ValueError, TypeError):
            supply_fmt = raw
        lines += [
            "",
            f"- Token Contract: `{contract}`",
            f"- Chain: {chain}",
            f"- On-Chain Circulating Supply: {supply_fmt}",
            "",
            "*Note: Holder count and tx volume require Etherscan Pro API.*",
        ]
    else:
        lines.append("\nOn-chain data unavailable for this token.")
    return "\n".join(lines)
