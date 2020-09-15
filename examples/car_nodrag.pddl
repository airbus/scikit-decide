(define (domain car)
(:requirements :typing :durative-actions :fluents :time :negative-preconditions :timed-initial-literals)

(:types t0 - object t1 - t0 t2 - (either t0 t1))

(:constants c0 - (either t0 t2) c1 - t1)

(:predicates (running ?g ?v - t0) (stopped) (engineBlown) (transmission_fine) (goal_reached) )

(:functions (d) (v) (a) (up_limit) (down_limit) (running_time) )

(:process moving
:parameters ()
:precondition (and (running))
:effect (and (increase (v) (* #t (a)))
             (increase (d) (* #t (v)))
	     (increase (running_time) (* #t 1))
)
)

(:action accelerate
  :parameters()
  :precondition (and (running) (< (a) (up_limit)))
  :effect (and (increase (a) 1))
)

(:action decelerate
  :parameters()
  :precondition (and (running) (> (a) (down_limit)))
  :effect (and (decrease (a) 1))
)

(:event engineExplode
:parameters ()
:precondition (and (running) (>= (a) 1) (>= (v) 100))
:effect (and (not (running)) (engineBlown) (assign (a) 0))
)

(:action stop
:parameters()
:precondition(and (= (v) 0) (>= (d) 30) (not (engineBlown)) )
:effect(goal_reached)
)

)

(define (problem car_prob)
    (:domain car)
	(:init
		(running)
		(transmission_fine)
		(= (running_time) 0)
		(= (up_limit) 1)
		(= (down_limit) -1)
		(= d 0)
		(= a 0)
		(= v 0)
	)
     (:goal (and (goal_reached) (not(engineBlown)) (<= (running_time) 50) (transmission_fine) ))
     (:metric minimize(total-time))
)
