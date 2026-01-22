# pip3 install undetected-chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

# # set up the driver in headless mode
# options = uc.ChromeOptions()
# options.add_argument("--headless=new")
# driver = uc.Chrome(options=options)

driver = uc.Chrome()

# navigate to the CreepJS test page
driver.get("https://www.sportmaster.ru/product/32500200299/")
time.sleep(1000)

# wait for the results to load
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "headless-resistance-detection-results"))
)

# take a screenshot of the results
element = driver.find_element(By.ID, "headless-resistance-detection-results")
element.screenshot("undetected_chromedriver_results.png")

driver.quit()
