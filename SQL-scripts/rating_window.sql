select customer_id,
title,
first_value(title) over rate as popular,
last_value(title) over rate as unpopular,
count(r.rental_id) over rate,
amount,rating
from inventory as i
join rental as r using(inventory_id) 
join payment as p using(customer_id)
join film as f using(film_id)
window rate as (partition by rating order by amount desc)
