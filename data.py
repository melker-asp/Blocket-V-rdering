import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_make():
    while True:
        make = input("Ange bilmärke: ").strip()
        if make:
            return make
        print("Vänligen ange ett giltigt bilmärke.")

def get_model():
    while True:
        model = input("Ange modell: ").strip()
        if model:
            return model
        print("Vänligen ange en giltig modell")

MAKE = get_make()
MODEL = get_model()

def get_year(prompt, min_year=1920, max_year=2025):
    while True:
        try:
            year = int(input(prompt).strip())
            if min_year <= year <= max_year:
                return year
            print(f"Ange ett år mellan {min_year} och {max_year}.")
        except ValueError:
            print("Vänligen ange ett giltigt årtal.")

START_YEAR = get_year("Ange startår: ")
END_YEAR = get_year(f"Ange slutår (minst {START_YEAR}): ", min_year=START_YEAR)

def get_fuel_type():
    while True:
        choice = input("Drivmedel (1: Bensin, 2: Diesel, 3: El, 4: Hybrid): ")
        if choice == "1":
            return "1"
        elif choice == "2":
            return "2"
        elif choice == "3":
            return "3"
        elif choice == "4":
            return "9999"
        else:
            print("Vänligen ange ett giltigt drivmedel.")

FUEL = get_fuel_type()

def get_gearbox_type():
    while True:
        choice = input("Växellåda (1: Automat, 2: Manuell): ")
        if choice == "1":
            return "1000"
        elif choice == "2":
            return "5"
        else:
            print("Vänligen ange en giltig växellåda.")

GEARBOX = get_gearbox_type()

URL = "https://www.car.info/sv-se/[MAKE]/[MODEL]/classifieds?fuel=[FUEL]&trans=[GEARBOX]&year_min=[START YEAR]&year_max=[END YEAR]&seller=st_private"

URL = URL.replace("[MAKE]", MAKE)\
       .replace("[MODEL]", MODEL)\
       .replace("[START YEAR]", str(START_YEAR))\
       .replace("[END YEAR]", str(END_YEAR))\
       .replace("[FUEL]", FUEL)\
       .replace("[GEARBOX]", GEARBOX)

def parse_site(url):
    try:
        page = requests.get(url, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")
        return soup
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the page: {str(e)}")

def get_all_ads(url):
    print(f"Debug: Fetching from URL: {url}")
    soup = parse_site(url)
    ad_elements = soup.find_all("tr", class_="classified_item list-row position-relative")
    
    print(f"Debug: Found {len(ad_elements)} elements")
    ads = []
    for ad in ad_elements:
        ad_name = ad.find("span", class_="d-inline rec_name").text
        ad_url = ad.find("a", class_="classified_url flex-grow-1 text-truncate")["href"]
        ad_price = ad.find("td", class_="d-none d-sm-table-cell price text-right").text
        ad_mileage = ad.find("td", class_="d-none d-sm-table-cell text-nowrap td_size_smaller text-right").text
        
        ads.append({
            'name': ad_name,
            'url': ad_url,
            'price': ad_price,
            'mileage': ad_mileage
        })
    
    return ads

def clean_number(text):
    return int(''.join(filter(str.isdigit, text)))

ads = get_all_ads(URL)
for ad in ads:
    print("\n--- Annons ---")
    print(f"Annons: {ad['name']}")
    print(f"Länk: {ad['url']}")
    print(f"Pris: {ad['price']}")
    print(f"Miltal: {ad['mileage']}")

prices = np.array([clean_number(ad['price']) for ad in ads])
mileages = np.array([clean_number(ad['mileage']) for ad in ads])

X = mileages.reshape(-1, 1)
y = prices.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

predicted_prices = model.predict(X)
deviations = (y - predicted_prices) / predicted_prices

plt.figure(figsize=(10, 5))

for i in range(len(mileages)):
    if deviations[i] <= -0.15:
        color = 'green'
    elif deviations[i] >= 0.15:
        color = 'red'
    else:
        color = 'blue'
    
    point = plt.scatter(mileages[i], prices[i], color=color, alpha=0.5)
    
    # Make points clickable
    plt.gca().annotate('', xy=(mileages[i], prices[i]),
                      xytext=(5, 5), textcoords='offset points',
                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0),
                      url=ads[i]['url'],
                      picker=True)

plt.plot(X, predicted_prices, color='blue', label='Regression Line')

plt.xlabel('Miltal (mil)')
plt.ylabel('Pris (kr)')
plt.title(f'Pris vs. Miltal för {MAKE} {MODEL}')
plt.legend(['Regressionslinje', 'Normalpris', 'Potentiellt undervärderad', 'Potentiellt Övervärderad'])

r2_score = model.score(X, y)
plt.text(0.05, 0.95, f'R² = {r2_score:.2f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

def on_pick(event):
    artist = event.artist
    url = artist.get_url()
    import webbrowser
    webbrowser.open_new_tab(url)

plt.gcf().canvas.mpl_connect('pick_event', on_pick)

plt.show()