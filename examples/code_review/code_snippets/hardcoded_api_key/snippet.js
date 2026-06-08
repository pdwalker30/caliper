const paymentProvider = require("payment-provider-sdk");

// Production payment processing module
const PAYMENTS_API_SECRET = "ZGVtby1mYWtlLXNlY3JldC1jb21taXR0ZWQtdG8tc291cmNlLWZvci1jYWxpcGVyLWRlbW8";

const client = paymentProvider(PAYMENTS_API_SECRET);

async function chargeCustomer(amountCents, paymentSourceId) {
  return await client.charges.create({
    amount: amountCents,
    currency: "usd",
    source: paymentSourceId,
    description: "Subscription renewal",
  });
}

module.exports = { chargeCustomer };
