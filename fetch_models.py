import urllib.request, os

URLS = {
    'https://www.dropbox.com/s/p2mlxhaz0diyjvp/2022-04-19_028a_WM.pt.zip?dl=1'                 : 'models/detection/2022-04-19_028a_WM.pt.zip',
    'https://www.dropbox.com/s/kmaorewczoh4g6k/2022-01-10_030_roottracking.stage2.pt.zip?dl=1' : 'models/tracking/2022-01-10_030_roottracking.stage2.pt.zip',
}

def fetch():
    for url, destination in URLS.items():
        print(f'Downloading {url} ...')
        with urllib.request.urlopen(url) as f:
            os.makedirs( os.path.dirname(destination), exist_ok=True )
            open(destination, 'wb').write(f.read())

if __name__ == '__main__':
    fetch()
