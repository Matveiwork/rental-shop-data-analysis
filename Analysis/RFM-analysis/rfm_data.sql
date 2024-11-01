
select  customer_id,
sum(amount) as amount,
count(customer_id) as n_buy,
max(to_char(payment_date, 'dd.mm.yyyy')) as last_buy
from customer
join payment using(customer_id)
group by customer_id
order by customer_id 