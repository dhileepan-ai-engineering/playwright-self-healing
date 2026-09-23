import { test } from '@playwright/test';
import { MarketingJournoPage } from '../pages/MarketingJournoPage';

test('Verify Marketing Journo Brand Display', async ({ page }) => {
  const mjPage = new MarketingJournoPage(page);

  await test.step('Navigate to Marketing Journo', async () => {
    await mjPage.navigate();
  });

  await test.step('Verify "MARKETING JOURNO" is displayed in the header', async () => {
    await mjPage.verifyBrandHeaderDisplayed();
  });
});