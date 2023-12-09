import requests
from bs4 import BeautifulSoup
import os

# 크롤링할 페이지 URL
url = "https://www.musinsa.com/mz/streetsnap"

# 해당 페이지의 HTML 가져오기
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# 이미지가 들어있는 <div class="articleImg"> 요소들 찾기
image_divs = soup.find_all("div", class_="articleImg")

# 이미지를 저장할 폴더 생성
os.makedirs("musinsa_images", exist_ok=True)

# 이미지 다운로드
for idx, image_div in enumerate(image_divs):
    img_url = "https:" + image_div.find("img")["src"]  # "https:" 스키마 추가
    img_name = f"musinsa_image_{idx+1}.jpg"
    img_path = os.path.join("musinsa_images", img_name)

    # 이미지 다운로드 요청
    img_data = requests.get(img_url).content

    # 이미지 저장
    with open(img_path, "wb") as img_file:
        img_file.write(img_data)

    print(f"이미지 '{img_name}' 다운로드 완료.")
