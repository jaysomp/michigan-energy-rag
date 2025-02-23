import sqlite3

def create_database():
    conn = sqlite3.connect("../energy_products.db")
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL
        )
    ''')
    
    # Create offers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS offers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            offer_details TEXT,
            validity TEXT,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
    ''')
    
     # 🔹 Create tickets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_request TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def seed_database():
    conn = sqlite3.connect("../energy_products.db")
    cursor = conn.cursor()
    
    # Sample product data
    products = [
        ("Green Energy Plan", "A renewable energy plan with 100% green electricity.", 49.99),
        ("Solar Panel Installation", "Professional solar panel installation for residential homes.", 4999.99),
        ("Fixed-Rate Electricity Package", "Lock in a fixed electricity rate for 24 months.", 79.99)
    ]
    
    # Insert products
    cursor.executemany("INSERT INTO products (name, description, price) VALUES (?, ?, ?)", products)
    
    # Fetch product IDs
    cursor.execute("SELECT id FROM products")
    product_ids = [row[0] for row in cursor.fetchall()]
    
    # Sample offers data
    offers = [
        (product_ids[0], "Get a 10% discount for the first 6 months", "2025-12-31"),
        (product_ids[1], "Free maintenance for the first year", "2025-06-30"),
        (product_ids[2], "$50 cashback for new customers", "2025-09-30")
    ]
    
    # Insert offers
    cursor.executemany("INSERT INTO offers (product_id, offer_details, validity) VALUES (?, ?, ?)", offers)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    seed_database()
    print("Database created and seeded successfully.")


