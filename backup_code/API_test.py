from questrade_api import get_account_info, search_symbol, get_quote

print("Searching AAPL...")
search_result = search_symbol("AAPL")
aapl_id = search_result['symbols'][0]['symbolId']
print("AAPL symbol ID:", aapl_id)

print("\nQuote:")
quote = get_quote(aapl_id)
print(quote)

print("\nAccount Info:")
print(get_account_info())
