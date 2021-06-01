-- Join select * from v2.active_user where domain_id in (X) and join to get subscriber_id by v2.active_user.id = v2.active_user_subscriber.user_id from v2.active_user_subscriber

-- wj 346938 woj 317642
--select count(*) from (
select * --v2.active_user.created_at au_created_at
--, v2.active_user_subscriber.created_at aus_created_at
--, * 
from 
v2.active_user
left join v2.active_user_subscriber
on v2.active_user.id = v2.active_user_subscriber.user_id
where -- updated_at != '-Infinity'
domain_id in ( 1274926143 , 308132225 ) --and channel_type_id <>2 and v2.active_user.id = 146685910
order by domain_id, v2.active_user.created_at desc
--) as a 
--
--where channel_type_id <>2
--limit 1000
