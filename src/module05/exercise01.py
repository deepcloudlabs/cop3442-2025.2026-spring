class OrderService:
    def __init__(self, tax_rate: float):
        self.tax_rate = tax_rate

    def calculate_total(self, price: float, quantity: int) -> float:
        if price < 0:
            raise ValueError("Price cannot be negative")

        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        subtotal = price * quantity
        tax = subtotal * self.tax_rate
        return subtotal + tax

    def format_order_summary(self, product_name: str, total: float) -> str:
        return f"Product: {product_name}, Total: {total:.2f}"

