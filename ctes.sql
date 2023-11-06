use md_water_services;
with 
audit as (SELECT
auditor_report.location_id AS audit_location,
auditor_report.true_water_source_score as audit_score,
-- visits.location_id AS visit_location,
auditor_report.type_of_water_source as audit_source,
visits.record_id as record_id,
visits.assigned_employee_id as employee
FROM
auditor_report
JOIN
visits
ON auditor_report.location_id = visits.location_id
where
	visits.visit_count = 1)
select
	audit.audit_location,
    audit.audit_score,
  --   audit.visit_location,
    ( select 
		employee_name
	from
		employee
	where
		employee.assigned_employee_id = audit.employee
    ) as employee_name,
    audit.record_id,
    water_quality.subjective_quality_score as employee_score
from
	audit
join
	water_quality
on audit.record_id = water_quality.record_id
where audit.audit_score != water_quality.subjective_quality_score
limit 10000;
