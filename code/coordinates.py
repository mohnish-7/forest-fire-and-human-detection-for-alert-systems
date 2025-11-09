def get_coordinates():

    import json
    from urllib.request import urlopen

    url = 'http://ipinfo.io/json'
    response = urlopen(url)
    data = json.load(response)

    ip = data['ip']
    coordinates = data['loc']
    coordinates = coordinates.split(',')
    lat = coordinates[0]
    long = coordinates[1]

    return ip, lat, long
