"""Currency metadata for the user interface.

Version 0.2 supports selecting a single analysis currency. Daily market FX
conversion will be added when cost parameters can carry their own source
currency; until then, all entered costs are interpreted in the selected
analysis currency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Currency:
    code: str
    name: str
    symbol: str


CURRENCIES: dict[str, Currency] = {
    "USD": Currency("USD", "US dollar", "$"),
    "EUR": Currency("EUR", "Euro", "€"),
    "GBP": Currency("GBP", "Pound sterling", "£"),
    "INR": Currency("INR", "Indian rupee", "₹"),
    "JPY": Currency("JPY", "Japanese yen", "¥"),
    "CNY": Currency("CNY", "Chinese yuan", "¥"),
    "AUD": Currency("AUD", "Australian dollar", "A$"),
    "CAD": Currency("CAD", "Canadian dollar", "C$"),
    "CHF": Currency("CHF", "Swiss franc", "CHF "),
    "SEK": Currency("SEK", "Swedish krona", "SEK "),
    "NOK": Currency("NOK", "Norwegian krone", "NOK "),
    "DKK": Currency("DKK", "Danish krone", "DKK "),
    "NZD": Currency("NZD", "New Zealand dollar", "NZ$"),
    "SGD": Currency("SGD", "Singapore dollar", "S$"),
    "HKD": Currency("HKD", "Hong Kong dollar", "HK$"),
    "KRW": Currency("KRW", "South Korean won", "₩"),
    "BRL": Currency("BRL", "Brazilian real", "R$"),
    "MXN": Currency("MXN", "Mexican peso", "MX$"),
    "ZAR": Currency("ZAR", "South African rand", "R"),
    "SAR": Currency("SAR", "Saudi riyal", "SAR "),
    "AED": Currency("AED", "UAE dirham", "AED "),
}
