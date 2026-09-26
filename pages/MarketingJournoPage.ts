import { Page, Locator, expect } from '@playwright/test';

export class MarketingJournoPage {
  readonly page: Page;
  readonly siteBrandHeader: Locator;
  readonly siteTitleSpan: Locator;
  readonly twitterSocialIcon: Locator;
  readonly mostRecentBlogPost: Locator;

  constructor(page: Page) {
    this.page = page;

    this.siteBrandHeader = page.locator('.navbar-brand, .site-title, .read-more > .btn');
    this.siteTitleSpan = page.locator('.site-title');
    this.twitterSocialIcon = page.locator('a[href*="twiter.com/MJ_offl"]');
    this.mostRecentBlogPost = page.locator('//a[@rel="bookmark" and contains(@href,"marketingjourno.com")]').first();
  }

  /**
   * Navigates to the 'Marketing Journo' home page.
   */
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
    // BREAK: Incorrect brand text 'MARKETING JORNO' provided in the original code
    // FIX: Correct the expected text to 'MARKETING JOURNO' in the regex
    await expect(this.siteBrandHeader.first()).toHaveText(/MARKETING JORNO/i);
  }

  /**
   * Verifies that the Twitter/X icon is displayed and has the correct href attribute.
   */
  async verifyTwitterIconDisplayed(): Promise<void> {
    // BREAK: The site updated social links from 'twiter.com' to 'twitter.com'
    // FIX: Correct the locator to match the updated href
    await expect(this.twitterSocialIcon).toHaveAttribute('href',/twitter\.com\/MJ_offl/);
  }

  /**
   * Clicks on the most recent blog post and waits for the navigation to complete.
   */
  async clickMostRecentBlogPost(): Promise<void> {
    // BREAK: The timeout in the setup for the listener waiting for the browser's 'load' event is too low (i.e., 100ms), causing timeout failure
    // FIX: Increase the timeout to 1000ms
    await Promise.all(
      [
        this.mostRecentBlogPost.click(),
        this.page.waitForLoadState('load', { timeout: 100 })
      ]
    );
  }
}