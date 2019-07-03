#ifndef AIRLAPS_PDDL_SYMBOL_CONTAINER_HELPER_HH
#define AIRLAPS_PDDL_SYMBOL_CONTAINER_HELPER_HH

namespace airlaps {

    namespace pddl {

        struct SymbolContainerHelper {
            template <typename Tholder, typename Tcontainer, typename Tsymbol>
            static const Tsymbol& add(const Tholder& holder, const std::string& holder_name,
                                      Tcontainer& container, const std::string& container_name,
                                      const Tsymbol& symbol, const std::string& symbol_name) {
                if (!container.emplace(symbol).second) {
                    throw std::logic_error("AIRLAPS exception: " + symbol_name + " '" +
                                           symbol->get_name() +
                                           "' already in the set of " + container_name + " of " + holder_name + " '" +
                                           holder->get_name() +
                                           "'");
                } else {
                    return symbol;
                }
            }

            template <typename Tholder, typename Tcontainer>
            static const typename Tcontainer::key_type& add(const Tholder& holder, const std::string& holder_name,
                                                            Tcontainer& container, const std::string& container_name,
                                                            const std::string& symbol, const std::string& symbol_name) {
                std::pair<typename Tcontainer::iterator, bool> i = container.emplace(std::make_shared<typename Tcontainer::key_type::element_type>(symbol));
                if (!i.second) {
                    throw std::logic_error("AIRLAPS exception: " + symbol_name + " '" +
                                           symbol +
                                           "' already in the set of " + container_name + " of " + holder_name + " '" +
                                           holder->get_name() +
                                           "'");
                } else {
                    return *i.first;
                }
            }

            template <typename Tholder, typename Tcontainer, typename Tsymbol>
            static void remove(const Tholder& holder, const std::string& holder_name,
                               Tcontainer& container, const std::string& container_name,
                               const Tsymbol& symbol, const std::string& symbol_name) {
                if (container.erase(symbol) == 0) {
                    throw std::logic_error("AIRLAPS exception: " + symbol_name + " '" +
                                           symbol->get_name() +
                                           "' not in the set of " + container_name + " of " + holder_name + " '" +
                                           holder->get_name() +
                                           "'");
                }
            }

            template <typename Tholder, typename Tcontainer>
            static void remove(const Tholder& holder, const std::string& holder_name,
                               Tcontainer& container, const std::string& container_name,
                               const std::string& symbol, const std::string& symbol_name) {
                remove(holder, holder_name,
                       container, container_name,
                       std::make_shared<typename Tcontainer::key_type::element_type>(symbol), symbol_name);
            }

            template <typename Tholder, typename Tcontainer>
            static const typename Tcontainer::key_type& get(const Tholder& holder, const std::string& holder_name,
                                                            Tcontainer& container, const std::string& container_name,
                                                            const std::string& symbol, const std::string& symbol_name) {
                typename Tcontainer::const_iterator i = container.find(std::make_shared<typename Tcontainer::key_type::element_type>(symbol));
                if (i == container.end()) {
                    throw std::logic_error("AIRLAPS exception: " + symbol_name + " '" +
                                           symbol +
                                           "' not in the set of " + container_name + " of " + holder_name + " '" +
                                           holder->get_name() +
                                           "'");
                } else {
                    return *i;
                }
            }
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_SYMBOL_CONTAINER_HELPER_HH
