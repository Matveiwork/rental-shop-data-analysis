#### На какую сумму делает покупки каждый клиент, в каждом рейтинговом сегменте,для каждого фильма,и сколько раз он совершает эту покупку.Так же,указана длина фильма,рейтинг и название.

select customer_id,
sum(amount) over choices ,
count(r.rental_id) over choices,
c.name as category,
length,rating,title
from inventory as i
join rental as r using(inventory_id) 
join payment as p using(customer_id)
join film as f using(film_id)
join film_category as fc using(film_id)
join category as c using(category_id)
window choices as (partition by customer_id,rating,title)

#### Какие фильмы продаются больше всего и меньше всего в разбиение по рейтингу.Так же,указана сумма уплаченная за взятие в аренду фильма,указан рейтинг и количество взятия в аренду опредленной рейтинговой группы.

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

#### На какое время обычно берут фильмы клиенты,сегментация клиентов с помощью квантилей по времени аренды.Так же,ууказано название фильма,рейтинг,категория фильма.

select customer_id,
extract(day from (return_date - rental_date)) as rental_period,
title,rating,c.name,
NTILE(3) OVER (PARTITION BY customer_id ORDER BY extract(day from (return_date - rental_date))) AS rental_period_quartile
from inventory as i
join rental as r using(inventory_id) 
join payment as p using(customer_id)
join film as f using(film_id)
join film_category as fc using(film_id)
join category as c using(category_id)

#### Создание единой витрины данных для визуализации. Выводит индентификатор клиента,сумму покупки,город,область,язык фильма,категория фильма,длина,рейтинговая категория,название,время аренды,фамилию сотрудника,год выпуска фильма,особенности фильма

with cus_add as (
select customer_id,amount,city,district,s.last_name,film_id,inventory_id
from address 
join city using(city_id)
join customer as cu using(address_id)
join inventory using(store_id)
join payment as p using(customer_id)
join staff as s using(staff_id)
)

select r.customer_id,amount,city,district,
c.name as category,length,rating,title,
extract(day from (return_date - rental_date)) as rental_period,
cad.last_name,l.name,release_year,special_features
from (select * from cus_add limit 100000) as cad
join film as f using(film_id)
join rental as r using(inventory_id)
join film_category as fc using(film_id)
join category as c using(category_id)
join language as l using(language_id)

#### Создание витрины данных для выявления сезонности в продажах по категориям. 
select name,amount,
extract(month from rental_date) as month_pay
from rental

join payment using(rental_id)
join inventory using(inventory_id)
join film_category using(film_id)
join category using(category_id)

#### Создание витрины данных для исследование какие актеры приносят больше прибыли,в каких городах какие актеры пользуются большей популярностью.(вложенный запрос с limit добавлен для оптимизации запроса,его можно исключить)

select a.last_name as actor,
sum(amount) over(partition by a.last_name) as profit,
first_value(city) over( partition by a.last_name order by amount desc) as popular_city
from (select * from payment limit 1000)
join customer using(customer_id)
join address using(address_id)
join city using(city_id)
join staff s using(staff_id)
join inventory i on s.store_id = i.store_id
join film_actor using(film_id)
join actor a using(actor_id)