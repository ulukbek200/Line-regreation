import { useQuery } from "@tanstack/react-query";
import { todoApi } from "./todoApi";

export const useTodos = () => {
  return useQuery({
    queryKey: ["todos"],
    queryFn: todoApi.getTodos,
  });
};