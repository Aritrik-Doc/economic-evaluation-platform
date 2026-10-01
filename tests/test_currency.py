from model.currency import CURRENCIES


def test_major_currency_catalog_contains_core_hta_markets():
    expected = {"USD", "EUR", "GBP", "INR", "JPY", "CNY", "AUD", "CAD", "CHF"}
    assert expected.issubset(CURRENCIES)


def test_currency_codes_match_objects():
    for code, currency in CURRENCIES.items():
        assert currency.code == code
        assert currency.name
        assert currency.symbol
