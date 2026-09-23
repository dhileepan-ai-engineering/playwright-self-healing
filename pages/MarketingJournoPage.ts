import { Page, Locator, expect } from '@playwright/test';

export class MarketingJournoPage {
  readonly page: Page;
  readonly siteBrandHeader: Locator;
  readonly siteTitleSpan: Locator;

  constructor(page: Page) {
    this.page = page;
    
    // Locators based on your outerHTML
    this.siteBrandHeader = page.locator('.navbar-brand, .site-title');
    this.siteTitleSpan = page.locator('.site-title');
  }

  async navigate(): Promise<void> {
    await this.page.goto('https://marketingjourno.com', { waitUntil: 'domcontentloaded' });
  }

  /**
   * Verifies that 'MARKETING JOURNO' is displayed in the header.
   */
  async verifyBrandHeaderDisplayed(): Promise<void> {
    // Ensure at least one header title element is visible
    await expect(this.siteBrandHeader.first()).toBeVisible({ timeout: 10000 });

    // Case-insensitive regex check for 'MARKETING JOURNO'
    await expect(this.siteBrandHeader.first()).toHaveText(/MARKETING JORNO/i);
  }
}