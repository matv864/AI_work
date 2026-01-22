# pip3 install undetected-chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchWindowException, WebDriverException
import clipboard
import keyboard
import winsound

# # set up the driver in headless mode
# options = uc.ChromeOptions()
# options.add_argument("--headless=new")
# driver = uc.Chrome(options=options)
chrome_path = None
options = uc.ChromeOptions()
options.page_load_strategy = 'eager'
driver = uc.Chrome(browser_executable_path=chrome_path, options=options)


def work_with_page(link: str) -> None:
    driver.get(link)
    try:
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.XPATH, "//h1[@class='mt-0']"))
        )
    except Exception:
        print("программа не смогла найти заголовок, дальше?????")
        winsound.Beep(2000, 3000)
        keyboard.wait('esc')
        return
    element = driver.find_element(By.XPATH, "//h1[@class='mt-0']")
    print(element.text)
    text_to_clipboard = f"{element.text} {link}"
    clipboard.copy(text_to_clipboard)
    print("дальше? ESC")
    [winsound.Beep(2000, 500) for _ in range(2)]
    keyboard.wait('esc')
   

with open("list.txt") as F:
    w = F.readlines()
for link in w:
    if "http" in link:
        print(link)
        work_with_page(link)
    else:
        [winsound.Beep(4000, 1000) for _ in range(7)]
        print(("!!!!!!!!!"*10 + "\n")*5)
        print("НОВАЯ ДАТА")
        print(link)

