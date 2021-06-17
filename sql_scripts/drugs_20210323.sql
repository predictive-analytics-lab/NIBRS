select
--	ofr.offender_id as offender_id,
--	ofr.incident_id as incident_id,
	case
		when ofr.ethnicity_id = 1 then 'hispanic/latino'
		when ofr.race_id = 1 then 'white'
		when ofr.race_id = 2 then 'black'
		when ofr.ethnicity_id is null or ofr.race_id = 0 or ofr.ethnicity_id = 3 then 'unknown'
		else 'other/mixed'
	end as dm_offender_race_ethnicity,
	case
		when ofr.sex_code = 'M' then 'male'
		when ofr.sex_code = 'F' then 'female'
		else 'unknown'
	end as dm_offender_sex,
	case
		when ofr.age_num < 18 then '12-17'
		when ofr.age_num < 26 then '18-25'
		when ofr.age_num < 35 then '26-34'
		when ofr.age_num < 50 then '35-49'
		when ofr.age_num >= 50  then '50+'
		else 'unknown'
	end as dm_offender_age,
--	ofr.offender_seq_num,
	i.cleared_except_id,
	coalesce(nat.arrest_type_name, 'No Arrest') arrest_type,
--	ag.state_abbr,
	concat(ag.pub_agency_name, '-', ag.state_name) as agency,
--	inc.drug_offense,
--	inc.drug_equipment_offense,
--	sus_drug.unique_drug_type_count,
--	sus_drug.drug_type_count,
--	sus_drug.unique_drug_measure_per_drug,
	sus_drug.crack_mass,
	sus_drug.cocaine_mass,
	sus_drug.heroin_mass,
	sus_drug.cannabis_mass,
	sus_drug.meth_amphetamines_mass,
	sus_drug.other_drugs,
	ca.criminal_act_count,
	ca.criminal_act,
	ca.criminal_act_id,
	inc_counts.offense_count,
	case
		when inc_counts.offender_count < 4 then cast(inc_counts.offender_count as varchar(1))
		else '4+'
	end as offender_count,
	inc_counts.location_category,
	case 
		when prop.drug_equipment_value = 0 then '0'
		when prop.drug_equipment_value < 100 then '1-100'
		when prop.drug_equipment_value < 1000 then '101-1000'
		else '1000+'
	end as drug_equipment_value
from nibrs_offender ofr
left join nibrs_incident i on ofr.incident_id = i.incident_id
left join nibrs_arrestee a on ofr.incident_id = a.incident_id and ofr.offender_seq_num = a.arrestee_seq_num
left join (
	select 
		count(prop.property_id) as property_count,
		prop.incident_id,
		sum(case
			when npd.prop_desc_id = 11 then npd.property_value
		end) as drug_equipment_value
	from nibrs_property prop
	left join nibrs_property_desc npd on prop.property_id = npd.property_id
	group by prop.incident_id
) prop on ofr.incident_id = prop.incident_id
INNER JOIN (
	SELECT
		o.incident_id,
		max(case 
			when o.offense_type_id = 16 then 1
			else 0
		end) as drug_offense,
		max(case
			when o.offense_type_id = 35 then 1
			else 0
		end) as drug_equipment_offense,
		max(case
			when o.offense_type_id not in (16, 35) then 1
			else 0
		end) as other_offense
	FROM nibrs_offense o
	group by o.incident_id
	) inc ON inc.incident_id = i.incident_id
left join (
	select
		COUNT(distinct sd.suspected_drug_type_id) as unique_drug_type_count,
		COUNT(distinct case
			when sd.suspected_drug_type_id = 5 then sd.drug_measure_type_id
			end) as unique_drug_measure_per_drug,
		COUNT(sd.suspected_drug_type_id) as drug_type_count,
		-- split quantities of 5 most common drugs into new columns
		sum(case
			when sd.suspected_drug_type_id = 1 then tot_mass.est_drug_mass
			else 0
		end) as crack_mass,
		sum(case
			when sd.suspected_drug_type_id = 2 then tot_mass.est_drug_mass
			else 0
		end) as cocaine_mass,
		sum(case
			when sd.suspected_drug_type_id = 4 then tot_mass.est_drug_mass
			else 0
		end) as heroin_mass,
		sum(case
			when sd.suspected_drug_type_id = 5 then tot_mass.est_drug_mass
			else 0
		end) as cannabis_mass,
		sum(case
			when sd.suspected_drug_type_id = 12 then tot_mass.est_drug_mass
			else 0
		end) as meth_amphetamines_mass,
		-- bool for other drugs being present too
		count(distinct case
			when sd.suspected_drug_type_id not in (1,2,4,5,12) then sd.suspected_drug_type_id
		end) as other_drugs,
		-- check if any quantities for the top drugs are in non-mass units
		count(case
			when sd.suspected_drug_type_id in (1,2,4,5,12) and sd.drug_measure_type_id not in (1, 2, 3, 4) then 1
		end) as non_mass_units,
		np.incident_id as incident_id
	FROM nibrs_property np
	    INNER JOIN nibrs_suspected_drug sd
	          ON sd.property_id = np.property_id
	    left join (
	    	select
	    		np.incident_id,
	    		sum(case
	    			when sd.drug_measure_type_id = 1 then sd.est_drug_qty			-- g
		    		when sd.drug_measure_type_id = 2 then sd.est_drug_qty * 1000	-- kg
		    		when sd.drug_measure_type_id = 3 then sd.est_drug_qty * 28.3495 -- oz
		    		when sd.drug_measure_type_id = 4 then sd.est_drug_qty * 453.592 -- lb
	    			else 0
	    		end) as est_drug_mass
	    	from nibrs_suspected_drug sd
	    	full outer join nibrs_property np on np.property_id = sd.property_id
	    	group by np.incident_id
	    ) tot_mass on np.incident_id = tot_mass.incident_id
	group by np.incident_id 
) sus_drug on sus_drug.incident_id = ofr.incident_id
left join (
	select
		COUNT(distinct ca1.criminal_act_id) as criminal_act_count,
		substring(max(distinct case
			when ca1.criminal_act_id = 8 then '0:consuming'
			when ca1.criminal_act_id = 6 then '1:possessing'
			when ca1.criminal_act_id = 1 then '2:buying'
			when ca1.criminal_act_id = 7 then '3:transporting'
			when ca1.criminal_act_id in (2, 5) then '4:producing'
			when ca1.criminal_act_id = 3 then '5:distributing'
		end), 3) as criminal_act,
		count(case when ca1.criminal_act_id not in (1,2,3,5,6,7,8) then 1 end) as other_criminal_acts,
		max(ca1.criminal_act_id) as criminal_act_id,
		ofr.incident_id
	FROM nibrs_offender ofr
		join nibrs_incident ni on ni.incident_id = ofr.incident_id
		join nibrs_offense no2 on no2.incident_id = ni.incident_id
		join nibrs_criminal_act ca1 on ca1.offense_id = no2.offense_id 
	group by ofr.incident_id 
) ca on ca.incident_id = i.incident_id
left join (
	select
		count(distinct no4.offender_id) as offender_count,
		count(distinct no3.offense_id) as offense_count,
		string_agg(distinct case
			when no3.location_id in (13, 18) then 'street'
			when no3.location_id in (8, 7, 23, 12) then 'store'
			when no3.location_id = 20 then 'home'
			when no3.location_id = 14 then 'hotel/motel'
			when no3.location_id = 41 then 'elementary school'
			else 'other'
		end,
		';') as location_category,
		count(distinct case
			when no3.location_id in (13, 18) then 'street'
			when no3.location_id in (8, 7, 23, 12) then 'store'
			when no3.location_id = 20 then 'home'
			when no3.location_id = 14 then 'hotel/motel'
			when no3.location_id = 41 then 'elementary school'
			else 'other'
		end) as location_count,
		ni.incident_id
	from nibrs_incident ni
		join nibrs_offense no3 on no3.incident_id = ni.incident_id
		join nibrs_offender no4 on no4.incident_id = ni.incident_id 
	group by ni.incident_id
) inc_counts on ofr.incident_id = inc_counts.incident_id
left join agencies ag on i.agency_id = ag.agency_id
left join nibrs_arrest_type nat on nat.arrest_type_id = a.arrest_type_id
where other_offense = 0 and inc_counts.location_count = 1 and sus_drug.non_mass_units = 0 and ca.other_criminal_acts = 0 and (ofr.ethnicity_id = 1 or ofr.race_id in (1, 2)) and ofr.sex_code ~ '(M|F)' and ofr.age_num is not null and ofr.age_num > 11










