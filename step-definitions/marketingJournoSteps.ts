import { Given, Then, Before, After } from '@cucumber/cucumber';
import { chromium, Browser, Page } from '@playwright/test';
import { MarketingJournoPage } from '../pages/MarketingJournoPage';

let browser: Browser;
let page: Page;
let mjPage: MarketingJournoPage;

Before(async () => {
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  page = await context.newPage();
  mjPage = new MarketingJournoPage(page);
});

After(async () => {
  await page.close();
  await browser.close();
});

Given('I navigate to the Marketing Journo home page', async () => {
  await mjPage.navigate();
});

Then('I should see {string} displayed in the header', async (expectedBrandText: string) => {
  await mjPage.verifyBrandHeaderDisplayed();
});