import pytest

from module05.exercise01 import OrderService

@pytest.fixture
def order_service() -> OrderService:
    return OrderService(tax_rate=0.2)

@pytest.mark.parametrize("price,quantity,total_order",[
    (100,5,600),
    (100,2,240),
    (100,1,120)
])
def test_calculate_total_should_be_successful(order_service,price,quantity,total_order) -> None:
    # step 2: call exercise method/function
    total_order = order_service.calculate_total(price=price,quantity=quantity)
    # step 3: verification
    assert total_order == total_order
    # step 4: tear-down test

@pytest.mark.parametrize("price,quantity",[
    (-1,0),
    (0,-1),
    (0,0)
])
def test_calculate_total_should_raise_error(order_service,price,quantity) -> None:
    # step 2 and 3: call exercise method/function + verification
    with pytest.raises(ValueError):
        order_service.calculate_total(price=price,quantity=quantity)
    # step 4: tear-down test