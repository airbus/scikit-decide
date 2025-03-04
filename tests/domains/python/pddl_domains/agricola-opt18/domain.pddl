;;Simulation of a simplified single player version of 'Agricola' board game
;;
;;Author: Tomas de la Rosa
;;        Universidad Carlos III of Madrid (2017)
;;
(define (domain agricola)
(:requirements :typing :negative-preconditions :action-costs)
(:types
    actiontag goods stage round worker improvement roundclass phaseclass roundparts resource room num - object
    buildtag animaltag vegtag gentag - actiontag
    animal vegetable - goods
)
(:constants
    num0 - num
    noworker - worker
    tnormal tharvest - roundclass
    harvest_init harvest_feeding harvest_breeding harvest_end - phaseclass
    sheep boar cattle - animal
    grain carrot - vegetable
    wood clay reed stone - resource
    act_rest act_labor act_plow act_build act_family act_sow act_fences act_improve void - gentag
    act_wood act_clay act_reed act_stone - buildtag
    oven fireplace - improvement
    act_grain act_carrot - vegtag
    act_sheep act_boar act_cattle - animaltag
    backhome renew roundend - roundparts
)
(:predicates
    (NEXT_STAGE ?s1 ?s2 - stage)
    (current_stage ?s - stage)
    (harvest_phase ?s - stage ?hclass - phaseclass)
    (NEXT_ROUND ?r1 ?r2 - round)
    (hold_round ?r - round ?p - roundparts)
    (current_round ?r - round)
    (CATEGORY_ROUND ?r - round ?t - roundclass)
    ;; Family members will be used in descending order
    ;; the max is the number of member at present
    (NEXT_WORKER ?w1 ?w2 - worker)
    (current_worker ?w - worker)
    (max_worker ?w - worker)
    (newborn)
    (plowed_fields)
    (stored_veg ?v - vegetable)
    (sown_veg ?v - vegetable)
    (can_harvest ?v - vegetable)
    (fences_for ?a - animal)
    (owned_animals ?a - animal)
    (can_breed ?a - animal)
    (NEXT_NUM ?i1 ?i2 - num)
    (NEXT2_NUM ?i1 ?i2 - num)
    (NUM_SUBSTRACT ?it - num ?iminus - num ?isol - num)
    (FOOD_REQUIRED ?w - worker ?i - num)
    (open_action ?a - actiontag)
    (available_action ?a - actiontag)
    (DRAWCARD_ROUND ?a - actiontag ?r - round)
    (num_food ?i - num)
    (stored_resource ?r - resource)
    (SUPPLY_RESOURCE ?s - buildtag ?r - resource)
    (space_rooms ?r - room)
    (built_rooms ?r - room ?w - worker)
    (ok)
    (home_improvement ?imp - improvement)
)
(:functions
    (total-cost) - number
    (group_worker_cost ?w - worker) - number
)
(:action ag__harvest_collect_end
    :parameters (?r - round ?s - stage)
    :precondition
    (and
        (hold_round ?r roundend)
        (harvest_phase ?s harvest_init)
        (CATEGORY_ROUND ?r tharvest)
    )
    :effect
    (and
        (not (harvest_phase ?s harvest_init))
        (harvest_phase ?s harvest_feeding)
        (increase (total-cost) 1)
    )
)

(:action ag__harvest_collecting_veg
    :parameters (?r - round ?s - stage ?v - vegetable ?i1 ?i2 ?i3 - num)
    :precondition
    (and
        (hold_round ?r roundend)
        (harvest_phase ?s harvest_init)
        (CATEGORY_ROUND ?r tharvest)
        (sown_veg ?v)
        (num_food ?i1)
        (NEXT2_NUM ?i1 ?i2)
        (NEXT_NUM ?i2 ?i3)
        (can_harvest ?v)
    )
    :effect
    (and
        (not (num_food ?i1))
        (num_food ?i3)
        (not (sown_veg ?v))
        (not (can_harvest ?v))
        (increase (total-cost) 1)
    )
)

;; This give an extra food is you have an oven
(:action ag__harvest_collecting_fromoven
    :parameters (?r - round ?s - stage ?v - vegetable ?i1 ?i2 ?i3 - num)
    :precondition
    (and
        (hold_round ?r roundend)
        (harvest_phase ?s harvest_init)
        (CATEGORY_ROUND ?r tharvest)
        (home_improvement oven)
        (sown_veg ?v)
        (num_food ?i1)
        (NEXT2_NUM ?i1 ?i2)
        (NEXT2_NUM ?i2 ?i3)
        (can_harvest ?v)
    )
    :effect
    (and
        (not (num_food ?i1))
        (num_food ?i3)
        (not (sown_veg ?v))
        (not (can_harvest ?v))
        (increase (total-cost) 1)
    )
)

(:action ag__harvest_feed
    :parameters (?r - round ?s - stage ?wmax - worker ?inow ?ifeed ?irest - num)
    :precondition
    (and
        (hold_round ?r roundend)
        (harvest_phase ?s harvest_feeding)
        (CATEGORY_ROUND ?r tharvest)
        (max_worker ?wmax)
        (FOOD_REQUIRED ?wmax ?ifeed)
        (num_food ?inow)
        (NUM_SUBSTRACT ?inow ?ifeed ?irest)
    )
    :effect
    (and
        (not (harvest_phase ?s harvest_feeding))
        (harvest_phase ?s harvest_breeding)
        (not (num_food ?inow))
        (num_food ?irest)
        (increase (total-cost) 1)
        (can_breed sheep)
        (can_breed boar)
        (can_breed cattle)
    )
)

(:action ag__harvest_breeding_animal
    :parameters (?r - round ?s - stage ?a - animal ?i ?i2 - num)
    :precondition
    (and
        (hold_round ?r roundend)
        (harvest_phase ?s harvest_breeding)
        (CATEGORY_ROUND ?r tharvest)
        (owned_animals ?a)
        (num_food ?i)
        (NEXT2_NUM ?i ?i2)
        (can_breed ?a)
    )
    :effect
    (and
        (not (num_food ?i))
        (num_food ?i2)
        (not (can_breed ?a))
        (increase (total-cost) 1)
    )
)

(:action ag__harvest_breed_end
    :parameters (?r - round ?s - stage)
    :precondition
    (and
        (hold_round ?r roundend)
        (harvest_phase ?s harvest_breeding)
        (CATEGORY_ROUND ?r tharvest)
    )
    :effect
    (and
        (not (harvest_phase ?s harvest_breeding))
        (harvest_phase ?s harvest_end)
        (increase (total-cost) 1)
    )
)

(:action ag__finish_round_backhome
    :parameters (?r - round ?maxw - worker)
    :precondition
    (and
        (current_round ?r)
        (current_worker noworker)
        (max_worker ?maxw)
        (not (newborn))
    )
    :effect
    (and
        (not (current_worker noworker))
        (current_worker ?maxw)
        (not (current_round ?r))
        (hold_round ?r backhome)
        (increase (total-cost) 1)
    )
)

(:action ag__finish_round_backhome_withchild
    :parameters (?r - round ?maxw ?newmax - worker)
    :precondition
    (and
        (current_round ?r)
        (current_worker noworker)
        (max_worker ?maxw)
        (NEXT_WORKER ?newmax ?maxw)
        (newborn)
    )
    :effect
    (and
        (not (current_worker noworker))
        (current_worker ?newmax)
        (not (max_worker ?maxw))
        (max_worker ?newmax)
        (not (current_round ?r))
        (not (newborn))
        (hold_round ?r backhome)
        (increase (total-cost) 1)
    )
)

(:action ag__finish_round_renew
    :parameters (?r - round ?maxw - worker)
    :precondition
    (and
        (hold_round ?r backhome)
    )
    :effect
    (and
        (not (hold_round ?r backhome))
        (hold_round ?r roundend)
        (available_action act_rest)
        (available_action act_labor)
        (available_action act_plow)
        (available_action act_grain)
        (available_action act_sow)
        (available_action act_sheep)
        (available_action act_wood)
        (available_action act_clay)
        (available_action act_stone)
        (available_action act_reed)
        (available_action act_family)
        (available_action act_build)
        (available_action act_fences)
        (available_action act_improve)
        (can_harvest grain)
        (can_harvest carrot)
        (increase (total-cost) 1)
    )
)

(:action ag__advance_round_normal
    :parameters (?r1 ?r2 - round ?act - actiontag)
    :precondition
    (and
        (CATEGORY_ROUND ?r1 tnormal)
        (hold_round ?r1 roundend)
        (NEXT_ROUND ?r1 ?r2)
        (DRAWCARD_ROUND ?act ?r2)
    )
    :effect
    (and
        (not (hold_round ?r1 roundend))
        (current_round ?r2)
        (open_action ?act)
        (increase (total-cost) 1)
    )
)

(:action ag__finish_stage
    :parameters (?s1 ?s2 - stage ?r1 ?r2 - round ?act - actiontag)
    :precondition
    (and
        (CATEGORY_ROUND ?r1 tharvest)
        (hold_round ?r1 roundend)
        (harvest_phase ?s1 harvest_end)
        (current_stage ?s1)
        (NEXT_STAGE ?s1 ?s2)
        (NEXT_ROUND ?r1 ?r2)
        (DRAWCARD_ROUND ?act ?r2)
    )
    :effect
    (and
        (not (hold_round ?r1 roundend))
        (not (current_stage ?s1))
        (current_round ?r2)
        (current_stage ?s2)
        (harvest_phase ?s2 harvest_init)
        (open_action ?act)
        (increase (total-cost) 1)
    )
)

;; ================================
;; PLAYER ACTIONS
;; ================================
(:action take_food
    :parameters (?w1 ?w2 ?wmax - worker ?r - round ?i1 ?i2 - num)
    :precondition
    (and
        (available_action act_labor)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (num_food ?i1)
        (NEXT_NUM ?i1 ?i2)
    )
    :effect
    (and
        (not (available_action act_labor))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (not (num_food ?i1))
        (num_food ?i2)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action plow_field
    :parameters (?w1 ?w2 ?wmax - worker ?r - round)
    :precondition
    (and
        (available_action act_plow)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (not (plowed_fields))
    )
    :effect
    (and
        (not (available_action act_plow))
        (plowed_fields)
        (not (current_worker ?w1))
        (current_worker ?w2)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action take_grain
    :parameters (?w1 ?w2 ?wmax - worker ?r - round ?v - vegetable)
    :precondition
    (and
        (available_action act_grain)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
    )
    :effect
    (and
        (not (available_action act_grain))
        (stored_veg ?v)
        (not (current_worker ?w1))
        (current_worker ?w2)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action build_fences
    :parameters (?a - animal ?w1 ?w2 ?wmax - worker ?r - round)
    :precondition
    (and
        (available_action act_fences)
        (open_action act_fences)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
    )
    :effect
    (and
        (not (available_action act_fences))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (fences_for ?a)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

;; Getting sheep, boar and cattle
(:action collect_animal
    :parameters (?a - animal ?act - animaltag ?w1 ?w2 ?wmax - worker ?r - round)
    :precondition
    (and
        (available_action ?act)
        (open_action ?act)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (fences_for ?a)
    )
    :effect
    (and
        (not (available_action ?act))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (owned_animals ?a)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

;; Getting sheep, boar and cattle
(:action collect_cook_animal
    :parameters (?a - animal ?act - animaltag ?w1 ?w2 ?wmax - worker ?r - round ?i1 ?i2 - num)
    :precondition
    (and
        (available_action ?act)
        (open_action ?act)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (home_improvement fireplace)
        (num_food ?i1)
        (NEXT2_NUM ?i1 ?i2)
    )
    :effect
    (and
        (not (available_action ?act))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (not (num_food ?i1))
        (num_food ?i2)
        (ok)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action collect_resource
    :parameters (?w1 ?w2 ?wmax - worker ?r - round ?act - buildtag ?res - resource)
    :precondition
    (and
        (available_action ?act)
        (open_action ?act)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (SUPPLY_RESOURCE ?act ?res)
    )
    :effect
    (and
        (not (available_action ?act))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (stored_resource ?res)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action build_room
    :parameters (?w1 ?w2 ?wmax ?wnewmax - worker ?r - round ?room - room)
    :precondition
    (and
        (available_action act_build)
        (open_action act_build)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (NEXT_WORKER ?wnewmax ?wmax)
        (current_round ?r)
        (stored_resource wood)
        (stored_resource reed)
        (space_rooms ?room)
    )
    :effect
    (and
        (not (available_action act_build))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (not (space_rooms ?room))
        (built_rooms ?room ?wnewmax)
        (not (stored_resource wood))
        (not (stored_resource reed))
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action improve_home
    :parameters (?w1 ?w2 ?wmax - worker ?r - round ?imp - improvement)
    :precondition
    (and
        (available_action act_improve)
        (open_action act_improve)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (stored_resource clay)
        (stored_resource stone)
    )
    :effect
    (and
        (not (available_action act_improve))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (home_improvement ?imp)
        (not (stored_resource clay))
        (not (stored_resource stone))
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action family_growth
    :parameters (?w1 ?w2 ?wmax ?wnewmax - worker ?r - round ?res - resource ?room - room)
    :precondition
    (and
        (available_action act_family)
        (open_action act_family)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (NEXT_WORKER ?wnewmax ?wmax)
        (built_rooms ?room ?wnewmax)
        (current_round ?r)
    )
    :effect
    (and
        (not (available_action act_family))
        (not (current_worker ?w1))
        (current_worker ?w2)
        (newborn)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

(:action sow
    :parameters (?w1 ?w2 ?wmax - worker ?r - round ?v - vegetable)
    :precondition
    (and
        (available_action act_sow)
        (open_action act_sow)
        (current_worker ?w1)
        (NEXT_WORKER ?w1 ?w2)
        (max_worker ?wmax)
        (current_round ?r)
        (plowed_fields)
        (stored_veg ?v)
    )
    :effect
    (and
        (not (available_action act_plow))
        (not (stored_veg ?v))
        (sown_veg ?v)
        (not (current_worker ?w1))
        (current_worker ?w2)
        (increase (total-cost) (group_worker_cost ?wmax))
    )
)

)
