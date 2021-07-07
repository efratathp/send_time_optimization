-- Run in DBeaver
--select count(*) -- disable for 
--from (
with relevant_notifications as (
	select 
		(select name from service.active_domains ad where id = domain_id) as "domain_name", 
		domain_id, id, 
		send_at, 
--		GET DOW OF SEND
--		trunc('hour', send_at), GET HOUR OF SEND
		date_part('hour', send_at) ts_hour,
		date_part('minute', send_at) ts_min,
		date_part('doy', send_at) ts_dayofyear,
		date_part('dow', send_at) ts_dayofweek,
		sent, unique_impressions, unique_opened,
		(unique_opened::decimal / sent::decimal * 100) as "unique_ctr"
		, trigger_type_id, channel_type_id --= 2 --web_push only
		, notification_status_type_id
	from v2.active_notification an 
	where send_at > now() - interval '190 day'
	and send_at < now() - interval '30 min'
	and domain_id in (
		select id from service.active_domains ad2 where name in (
			'joy.land', 'masa.co.il', 'westernjournal.com' -- TODO change this to param domain_id
		)
	) 
	and trigger_type_id in (
		select id from v3.trigger_type where type in (
			'dispatcher_scheduled_personalized_notification'
		)	
	)	
	and channel_type_id = 2 --web_push only
	and notification_status_type_id = 3 --successfully sent
)
, notifications_grouped_by_dow_and_hour as (
	select *
	--	domain_id, avg(unique_ctr)
	from relevant_notifications
--	group by domain_id, (NEED TO GROUP BY TRUNCATED HOUR + DOW)
)
select * 
from relevant_notifications
order by domain_id asc, send_at desc
--) as f