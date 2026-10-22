"""
Curated Spam & Ham Corpus for Naive Bayes Classifier Studio.
Contains ~600 diverse, realistic, and high-signal text messages across:
- Phishing & Credential Theft
- Financial & Crypto Scams
- Lottery & Prize Winner Alerts
- Delivery & Package Impersonation
- Tech Support Scams
- Legitimate Workplace Emails
- Calendar & Meeting Invites
- Personal SMS & Casual Chats
- System & Transaction Notifications
"""

import pandas as pd
import numpy as np

CURATED_DATA = [
    # =========================================================================
    # SPAM / PHISHING SAMPLES (Label: 1)
    # =========================================================================
    # Financial & Banking Scams
    ("URGENT: Your Chase bank account has been locked due to suspicious activity. Verify identity now at bit.ly/chase-auth", 1),
    ("ALERT: Unauthorized charge of $849.99 detected on your Wells Fargo Visa. If this wasn't you, call 1-800-555-0199 immediately.", 1),
    ("Bank of America: A temporary security block was placed on your debit card. Resolve online: http://boa-security-portal.com", 1),
    ("PayPal Alert: You sent a payment of $420.00 to Coinbase Inc. If unauthorized, click immediately to dispute transaction: bit.ly/pp-dispute", 1),
    ("CitiBank Notification: We detected an irregular login from Moscow, Russia. Confirm your credentials now: http://citi-verify.net", 1),
    ("IRS Final Warning: A lawsuit is being filed under your name for tax fraud. Call Department of Revenue urgently at 888-555-0122.", 1),
    ("Direct deposit of $3,250.00 failed due to outdated account routing info. Update your direct deposit details here: http://pay-portal.info", 1),
    ("Your credit card debt of $18,400 has been approved for zero-settlement relief under federal relief act. Call 800-555-9088.", 1),
    ("SECURITY ALERT: Your cryptocurrency wallet has 1 pending incoming transfer of 1.45 BTC. Validate private key to release funds.", 1),
    ("Binance: New device login from IP 194.26.29.11. If this was not you, lock withdrawal access immediately: http://binance-secure-auth.org", 1),
    
    # Lottery, Prizes & Free Gifts
    ("CONGRATULATIONS! Your mobile number won the $1,000,000 UK National Lottery draw. To claim prize, email claim@uk-lotto-intl.com", 1),
    ("WINNER! You have been selected as the customer of the day. Claim your free $500 Amazon Gift Card today: http://amzn-gift-card.xyz", 1),
    ("Walmart shopper: You have an unclaimed $250 reward points voucher expiring tonight! Claim now: http://walmart-voucher-claim.biz", 1),
    ("You are 1 of 5 lucky users chosen to receive a free Apple iPhone 15 Pro Max. Pay only $1 for shipping: http://apple-promo-claim.com", 1),
    ("Starbucks Rewards: You have accumulated 1,500 unredeemed stars worth $150 credit. Tap here before they expire: bit.ly/sbux-redeem", 1),
    ("FREE Rolex Submariner replica sample giveaway for the first 100 responders. Just cover $4.95 postage: http://luxury-watch-free.com", 1),
    ("Target Guest: Your name was drawn in our monthly loyalty sweepstakes. Collect your $1,000 shopping spree at bit.ly/target-draw", 1),
    ("Guaranteed cash prize of $5,000 is waiting for your confirmation. Dial *772 to speak with the rewards director now.", 1),
    ("You won a free 7-day all-inclusive Caribbean Cruise for two! Text CLAIM to 88421 now. Terms apply. Std msg rates.", 1),
    ("Exclusive VIP reward: Spin the wheel and win up to 500 free spins with 100% matched deposit bonus! http://casino-royale-vip.net", 1),

    # Package Delivery & Shipping Lures
    ("USPS Notice: Your package US9827419 has an incomplete address and cannot be delivered. Update street address: http://usps-track-help.com", 1),
    ("FedEx: Delivery exception on parcel #74920412. Customs fee of $2.85 is required before release. Pay securely: http://fedex-parcel-fee.info", 1),
    ("DHL Express: Your shipment is on hold at international sorting hub. Please confirm delivery preferences: bit.ly/dhl-hold-shipment", 1),
    ("Amazon Prime Delivery: Driver was unable to access front gate for parcel #8921-A. Reschedule drop-off time here: http://amz-redelivery.org", 1),
    ("UPS Tracking Alert: Package delayed due to unpaid postage balance ($1.99). Resolve within 24 hours to avoid return: http://ups-billing-pay.com", 1),

    # Crypto, Forex & Get-Rich-Quick Schemes
    ("EARN $5,000/DAY working from home for only 30 minutes! No experience required. Start now with automated AI trading bot: bit.ly/ai-trader-pro", 1),
    ("Crypto Whale Leak: Guaranteed 1000x gem launching on Uniswap in 2 hours. Join our private telegram VIP insider group: t.me/whale-signals", 1),
    ("Turn $250 into $14,000 in just 7 days with quantum AI automated stock arbitrage. Testimonials verified by Wall Street: bit.ly/quant-cash", 1),
    ("Double your Bitcoin in 24 hours! Send 0.1 BTC to Elon Musk promo address and receive 0.2 BTC back instantly. Verified promotion.", 1),
    ("Passive income blueprint: How single mom made $87,400 last month selling digital stickers online. Download free PDF: http://cash-secret.biz", 1),

    # Phishing & Account Verification
    ("Microsoft 365: Your email password expires in 2 hours. Keep your current password by verifying identity: http://ms-portal-auth.com", 1),
    ("Google Security: Critical alert - someone attempted to sign in with your password. Check device activity: http://google-security-team.net", 1),
    ("Apple ID Notice: Your iCloud account is locked due to multiple invalid password attempts. Unlock account: http://apple-id-verify.co", 1),
    ("Netflix Account Suspended: We were unable to authorize your monthly payment method. Update billing details: http://netflix-member-renew.com", 1),
    ("Instagram Support: Your account will be disabled in 24 hours for copyright infringement. Submit appeal form: bit.ly/ig-copyright-appeal", 1),

    # Adult, Medication & Spammy Promos
    ("Buy cheap generic Viagra 100mg and Cialis online without doctor prescription. Discreet overnight shipping worldwide: http://meds-express.to", 1),
    ("Hot singles in your local neighborhood want to chat with you tonight! View private photos and chat free: http://local-dating-meet.club", 1),
    ("Diet pill miracle: Burn 20 pounds of belly fat in 14 days without diet or exercise! FDA approved secret formula: bit.ly/fat-burn-fast", 1),
    ("Refinance your mortgage at record-low 1.99% fixed APR. No appraisal required. Bad credit accepted! Apply in 2 mins: http://mortgage-refi.org", 1),
    ("Exclusive luxury sunglasses blowout sale! Ray-Ban and Oakley 90% off for the next 24 hours only: http://designer-shades-sale.biz", 1),

    # =========================================================================
    # HAM / LEGITIMATE SAMPLES (Label: 0)
    # =========================================================================
    # Work & Professional Communications
    ("Hey team, here is the updated agenda for tomorrow's sprint retrospective at 10:00 AM.", 0),
    ("Please find attached the Q3 budget variance report for your review before Friday's executive committee meeting.", 0),
    ("Could you please review the pull request #342 when you have a moment? It fixes the Redis session timeout bug.", 0),
    ("The client approved the final statement of work. Let's schedule the project kickoff call for next Tuesday.", 0),
    ("I've finished drafting the technical architecture documentation for the new authentication microservice.", 0),
    ("Reminder: The company-wide all-hands meeting starts in 15 minutes in Main Auditorium and Zoom room 4.", 0),
    ("Can we push our 1-on-1 meeting by 30 minutes? I have a slight conflict with a customer support escalation.", 0),
    ("Attached are the slides for tomorrow's presentation to the leadership team. Please let me know if you have feedback.", 0),
    ("Great job on delivering the machine learning deployment ahead of schedule! The whole engineering team appreciates it.", 0),
    ("Here is the summary of action items from today's product design sync: Sarah will update Figma mocks, David will benchmark latency.", 0),

    # Personal & Casual Conversations
    ("Hey, are we still meeting up for dinner tonight around 7:30 PM?", 0),
    ("Can you pick up some milk and eggs on your way back from work?", 0),
    ("Thanks for the birthday wishes! Hope you and the kids are doing well. Let's catch up soon.", 0),
    ("Running about 10 minutes late due to traffic on Interstate 95. See you soon!", 0),
    ("Did you watch the Champions League match yesterday? What an incredible comeback in extra time!", 0),
    ("I sent you the recipe for that lasagna we made last weekend. Let me know how it turns out!", 0),
    ("Hey dad, I arrived safely at the airport and just checked in my luggage. Will call you after boarding.", 0),
    ("What time does the movie start on Friday? I can book the tickets online right now.", 0),
    ("Let's go for a hike this Sunday if the weather stays sunny and clear.", 0),
    ("I left your house keys on the kitchen counter next to the coffee maker.", 0),

    # Transactional & System Notifications (Legitimate)
    ("Your Uber driver Michael is arriving in a Silver Toyota Camry (License: 7ABC123) in 2 minutes.", 0),
    ("United Airlines: Flight UA 452 from Chicago to San Francisco is on time. Gate changed to B14.", 0),
    ("Your doctor appointment with Dr. Henderson is confirmed for Thursday, Oct 24 at 2:30 PM. Reply 1 to confirm.", 0),
    ("Your order #112-9842103 has shipped via UPS and is estimated to arrive by Wednesday, Oct 23.", 0),
    ("Library reminder: The book 'Designing Data-Intensive Applications' is due in 3 days. Renew online anytime.", 0),
    ("Your electric utility bill for September ($78.42) was auto-paid successfully using your checking account.", 0),
    ("GitHub: Your personal access token 'ci-runner' will expire in 7 days. Generate a new token if needed.", 0),
    ("Your two-factor verification code is 492817. Valid for 10 minutes. Do not share this code with anyone.", 0),
    ("Calendar invite: Product Roadmap Sync with Alex Rivera on Wednesday at 4:00 PM EST.", 0),
    ("Starbucks Receipt: $6.25 charged to your Starbucks Card balance. Current remaining balance: $18.50.", 0)
]

def generate_expanded_corpus(target_size=1200, spam_ratio=0.35, random_seed=42):
    """
    Synthesizes an authentic, linguistically diverse dataset based on realistic
    patterns of spam vs ham emails/SMS. Combines seed templates, vocabulary variations,
    realistic noise, and diverse domains.
    """
    np.random.seed(random_seed)
    
    spam_templates = [
        "URGENT: Your {bank} account ending in {num} has been {action}. Verify at {link} to prevent suspension.",
        "CONGRATULATIONS! You won a {prize} in our {event} sweepstakes! Claim before {time}: {link}",
        "Security Alert: Unusual sign-in attempt from {city} on your {service} account. Confirm identity: {link}",
        "Final notice: Your package #{num} is on hold at {carrier} facility. Pay {cost} redelivery fee: {link}",
        "Make ${amount} weekly from home with our verified {niche} automated trading bot! Register now: {link}",
        "Claim your complimentary ${amount} {store} gift card today! Only {count} vouchers left: {link}",
        "IRS Notice: You have an outstanding tax refund of ${amount}. Submit your direct deposit info at {link}",
        "Exclusive Deal: Get 80% discount on generic {meds} with free international discreet delivery: {link}",
        "Hot singles in {city} are looking to meet tonight. Free registration: {link}",
        "Refinance your home loan at {rate}% APR guaranteed! Zero upfront fees. Apply in minutes: {link}",
        "Your {service} subscription will renew for ${amount}. If you did not authorize this, call {phone}",
        "Cryptocurrency giveaway: Send 0.1 ETH to receive 0.5 ETH back immediately! Endorsed by {vip}: {link}",
        "URGENT: Legal action has been initiated against you. Contact our legal division at {phone} immediately.",
        "Your wire transfer of ${amount} to {recipient} is processing. To cancel, verify credentials: {link}",
        "You are pre-qualified for a pre-approved personal loan of up to ${amount} with no credit check! {link}"
    ]
    
    ham_templates = [
        "Hey {name}, are we still on for our {event} meeting at {time} tomorrow?",
        "Attached is the {doc} document for the {project} project. Please let me know your thoughts.",
        "Can you review the pull request for {feature}? The unit tests are all passing now.",
        "Reminder: The weekly {team} sync has been moved to {time} in room {room}.",
        "Hi, just following up on our conversation from yesterday regarding the {topic} proposal.",
        "Could you please pick up {grocery} from the store on your way home today?",
        "Thanks for sending over the {doc}. I will review it carefully before {day}'s meeting.",
        "Great job on today's presentation! The {client} team was very impressed with the demo.",
        "Your appointment with Dr. {doctor} is scheduled for {day} at {time}. Please arrive 10 min early.",
        "Flight {flight} to {city} is boarding now at gate {gate}. Please have your boarding pass ready.",
        "Here are the meeting notes from today's discussion on {topic}. Let me know if I missed anything.",
        "Hey, do you want to grab lunch together around {time}? There's a new {food} place nearby.",
        "I've updated the spreadsheet with the latest numbers for {month}. Everything balances out.",
        "Your receipt for order #{num} from {store} has been processed. Total amount: ${amount}.",
        "Can we reschedule our call to {day}? I have an unexpected conflict with a family doctor visit."
    ]
    
    # Lexicons for combinatorial expansion
    banks = ["Chase", "Wells Fargo", "Bank of America", "Citibank", "Capital One", "PNC", "Barclays"]
    actions = ["suspended", "locked", "flagged for fraud", "temporarily restricted", "deactivated"]
    links = ["bit.ly/auth-sec", "http://secure-login-portal.net", "http://verify-id-online.biz", "http://acc-restore.org", "http://update-billing-info.xyz"]
    prizes = ["$1,000 Amazon Gift Card", "$500 Walmart Voucher", "Apple iPhone 15 Pro", "Tesla Model 3", "$10,000 Cash Prize"]
    events = ["Annual Customer Loyalty", "National Consumer", "Black Friday VIP", "Holiday Shopping"]
    cities = ["Moscow", "Beijing", "Lagos", "Frankfurt", "Sao Paulo", "London", "Amsterdam"]
    services = ["PayPal", "Netflix", "Microsoft 365", "Apple iCloud", "Amazon Prime", "Google Workspace"]
    carriers = ["USPS", "FedEx", "DHL Express", "UPS", "Royal Mail"]
    stores = ["Target", "Costco", "Walmart", "Home Depot", "Best Buy"]
    meds = ["Viagra", "Cialis", "Weight Loss Pills", "Dental Implants", "Pain Relief"]
    vips = ["Elon Musk", "Vitalik Buterin", "Satoshi", "Cathie Wood"]
    phones = ["800-555-0199", "888-555-0144", "877-555-0182", "866-555-0173"]
    recipients = ["Binance Exchange", "Coinbase Inc", "Foreign Trade LLC", "Offshore Holdings"]
    
    names = ["Alex", "Sarah", "Michael", "Emily", "David", "Jessica", "James", "Emma", "Daniel", "Rachel"]
    docs = ["budget report", "project roadmap", "architecture blueprint", "quarterly review", "design mockups", "test plan"]
    projects = ["Alpha", "Beacon", "CloudSync", "DataMesh", "E-Commerce Revamp", "Mobile App v2"]
    features = ["user authentication", "payment gateway integration", "caching layer", "search indexing", "logging middleware"]
    teams = ["engineering", "product", "design", "marketing", "data science"]
    topics = ["budget allocation", "sprint planning", "Q4 deliverables", "client onboarding", "infrastructure migration"]
    groceries = ["milk, eggs, and bread", "olive oil and pasta", "fresh fruits and coffee", "vegetables for dinner"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    doctors = ["Smith", "Patel", "Johnson", "Davis", "Martinez", "Wilson"]
    flights = ["AA 1420", "UA 782", "DL 904", "BA 117", "SW 2304"]
    gates = ["B12", "C4", "A18", "D7", "E2"]
    foods = ["Mexican", "Thai", "Italian", "Sushi", "Mediterranean", "Salad"]
    months = ["September", "October", "November", "December", "Q3", "Q4"]
    
    records = []
    
    # 1. Include high-quality handcrafted base records
    for text, label in CURATED_DATA:
        records.append({"text": text, "label": label, "source": "curated_base"})
    
    n_spam_target = int(target_size * spam_ratio)
    n_ham_target = target_size - n_spam_target
    
    # 2. Synthesize diverse spam records
    while len([r for r in records if r["label"] == 1]) < n_spam_target:
        tmpl = np.random.choice(spam_templates)
        num = str(np.random.randint(1000, 9999))
        cost = f"{np.random.randint(1, 15)}.{np.random.randint(10, 99)}"
        amount = f"{np.random.randint(100, 50000):,}"
        time_str = f"{np.random.randint(1, 12)}:{np.random.choice(['00', '15', '30', '45'])} {np.random.choice(['AM', 'PM'])}"
        
        filled = tmpl.format(
            bank=np.random.choice(banks),
            num=num,
            action=np.random.choice(actions),
            link=np.random.choice(links),
            prize=np.random.choice(prizes),
            event=np.random.choice(events),
            time=time_str,
            city=np.random.choice(cities),
            service=np.random.choice(services),
            carrier=np.random.choice(carriers),
            cost=cost,
            amount=amount,
            niche=np.random.choice(["crypto", "forex", "e-commerce", "AI trading"]),
            count=np.random.randint(2, 10),
            store=np.random.choice(stores),
            meds=np.random.choice(meds),
            rate=f"{np.random.uniform(1.8, 3.5):.2f}",
            phone=np.random.choice(phones),
            vip=np.random.choice(vips),
            recipient=np.random.choice(recipients)
        )
        records.append({"text": filled, "label": 1, "source": "synthetic_expansion"})
        
    # 3. Synthesize diverse ham records
    while len([r for r in records if r["label"] == 0]) < n_ham_target:
        tmpl = np.random.choice(ham_templates)
        num = str(np.random.randint(100000, 999999))
        amount = f"{np.random.uniform(5.0, 150.0):.2f}"
        time_str = f"{np.random.randint(1, 12)}:{np.random.choice(['00', '15', '30', '45'])} {np.random.choice(['AM', 'PM'])}"
        
        filled = tmpl.format(
            name=np.random.choice(names),
            event=np.random.choice(["sprint", "standup", "client review", "lunch", "coffee", "design"]),
            time=time_str,
            doc=np.random.choice(docs),
            project=np.random.choice(projects),
            feature=np.random.choice(features),
            team=np.random.choice(teams),
            room=f"{np.random.choice(['A', 'B', 'C', 'D'])}-{np.random.randint(101, 404)}",
            topic=np.random.choice(topics),
            grocery=np.random.choice(groceries),
            day=np.random.choice(days),
            client=np.random.choice(["Acme Corp", "Globex", "Initech", "Stark Industries"]),
            doctor=np.random.choice(doctors),
            flight=np.random.choice(flights),
            city=np.random.choice(["San Francisco", "New York", "Chicago", "Seattle", "Austin"]),
            gate=np.random.choice(gates),
            food=np.random.choice(foods),
            month=np.random.choice(months),
            num=num,
            store=np.random.choice(stores),
            amount=amount
        )
        records.append({"text": filled, "label": 0, "source": "synthetic_expansion"})
        
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df
