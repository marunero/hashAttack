from pytineye import TinEyeAPIRequest

api_key = '6mm60lsCNIB,FwOWjJqA80QZHh9BMwc-ber4u=t^'
api = TinEyeAPIRequest(
    api_url = "https://api.tineye.com/rest/search",
    api_key = api_key,
)



with open('modified_img.png', 'rb') as fp:
    data = fp.read()
    response = api.search_data(
        data=data
    )

print(response)
print("hi")